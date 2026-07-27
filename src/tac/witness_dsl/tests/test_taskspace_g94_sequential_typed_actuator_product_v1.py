from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
    LanePeriodicProgramV1,
    MovableWorldsheetTrackV1,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (
    ConditionalY0ControlV1,
    PopulationConditionalOperandV1,
)
from tac.witness_dsl.taskspace_g89_class_complete_semantic_compiler_v1 import (
    ClassCompleteSemanticProgramV1,
    SharedTopologyApplicationV1,
    SharedTopologyTemplateV1,
)
from tac.witness_dsl.taskspace_g94_sequential_typed_actuator_product_v1 import (
    CAUSAL_TRANSITION_ID,
    CONDITIONAL_FIT_AUTHORITY,
    OPEN_BLOCKERS,
    PRODUCT_MAGIC,
    ParsedSequentialTypedProductV1,
    SequentialTypedArchiveBuildV1,
    SequentialTypedBatchResultV1,
    SequentialTypedProductError,
    SequentialTypedProductReceiverV1,
    build_g83_state_metadata,
    build_sequential_typed_archive,
    encode_sequential_typed_product,
    parse_sequential_typed_product,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    parse_taskspace_outer_archive,
)

CURRENT_BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/g85_pvsa_public_receiver_20260727_r1/archive.zip")
CURRENT_BASE_ARCHIVE_BYTES = 129_392
CURRENT_BASE_ARCHIVE_SHA256 = "b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd"
CURRENT_BASE_MEMBER_BYTES = 133_363
CURRENT_BASE_MEMBER_SHA256 = "d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31"
SEMANTIC_P_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"


def _current_base_member() -> bytes:
    archive = CURRENT_BASE_ARCHIVE.read_bytes()
    assert len(archive) == CURRENT_BASE_ARCHIVE_BYTES
    assert hashlib.sha256(archive).hexdigest() == CURRENT_BASE_ARCHIVE_SHA256
    parsed = parse_taskspace_outer_archive(
        archive,
        expected_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
    )
    assert len(parsed.member_bytes) == CURRENT_BASE_MEMBER_BYTES
    assert parsed.member_sha256 == CURRENT_BASE_MEMBER_SHA256
    return parsed.member_bytes


def _program(
    *,
    collide_with_incumbent: bool = False,
    lane_width_bias_q8: int = 64,
) -> ClassCompleteSemanticProgramV1:
    road = (
        BoundaryShearletAtomV1(0, "Road", 160, 256, 24, 96, 0, 64)
        if collide_with_incumbent
        else BoundaryShearletAtomV1(0, "Road", 240, 494, 4, 8, 0, 64)
    )
    return ClassCompleteSemanticProgramV1(
        semantic_archive_sha256=SEMANTIC_P_SHA256,
        topology_templates=(SharedTopologyTemplateV1(0, "birth", "box", 1, 3, 3),),
        topology_applications=(
            SharedTopologyApplicationV1(0, "UndrivableBoundary", 0, 0, 0),
            SharedTopologyApplicationV1(0, "Road", 0, 0, 5),
            SharedTopologyApplicationV1(0, "Lane", 0, 0, 10),
            SharedTopologyApplicationV1(0, "MyCar", 0, 0, 20),
            SharedTopologyApplicationV1(0, "Movable", 0, 0, 15),
        ),
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
            road,
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
                width_bias_q8=lane_width_bias_q8,
                width_slope_q12=0,
            ),
        ),
        lane_knots=(),
    )


def _conditional(base_member: bytes) -> PopulationConditionalOperandV1:
    return PopulationConditionalOperandV1(
        base_pvsa_member_sha256=hashlib.sha256(base_member).hexdigest(),
        semantic_p_sha256=SEMANTIC_P_SHA256,
        controls=(ConditionalY0ControlV1.copy_conditional_y1(0),),
    )


@pytest.fixture(scope="module")
def exact_fixture() -> tuple[
    bytes,
    SequentialTypedArchiveBuildV1,
    SequentialTypedProductReceiverV1,
    SequentialTypedBatchResultV1,
]:
    base = _current_base_member()
    program = _program()
    conditional = _conditional(base)
    build = build_sequential_typed_archive(
        base_pvsa_member_bytes=base,
        g89_program_bytes=program.to_bytes(),
        g88_conditional_operand_bytes=conditional.to_bytes(),
    )
    receiver = build.selected.open_receiver(verify_member_effects=False)
    result = receiver.decode_pair(0)
    return base, build, receiver, result


def test_exact_current_base_sequential_product_closes_all_three_transitions(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    base, build, _receiver, result = exact_fixture
    selected = build.outer_build.selected
    assert build.stored == build.deflated == build.selected
    assert build.selected.member_bytes.startswith(PRODUCT_MAGIC)
    assert build.selected.base_pvsa_member_bytes == base
    assert build.selected.base_pvsa.member_sha256 == CURRENT_BASE_MEMBER_SHA256
    assert build.selected.g89_program.semantic_archive_sha256 == SEMANTIC_P_SHA256
    assert build.selected.g88_conditional_operand.base_pvsa_member_sha256 == CURRENT_BASE_MEMBER_SHA256
    assert len(selected.archive_bytes) == selected.archive_nbytes
    assert hashlib.sha256(selected.archive_bytes).hexdigest() == selected.archive_sha256

    assert result.deterministic_double_decode is True
    assert result.conditional_result.deterministic_double_decode is True
    assert np.array_equal(
        result.preconditional_camera_pairs[:, 0],
        result.base_incumbent_camera_pairs[:, 0],
    )
    assert result.g89_changed_y1_values > 0
    assert np.array_equal(
        result.camera_pairs[:, 1],
        result.preconditional_camera_pairs[:, 1],
    )
    assert result.g88_changed_y0_values > 0
    # COPY_CONDITIONAL_Y1 makes conditional dependence executable.
    assert np.array_equal(
        result.camera_pairs[:, 0],
        result.preconditional_camera_pairs[:, 1],
    )


def test_source_specific_byte_homes_are_exact_and_nonduplicated(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    _base, build, _receiver, _result = exact_fixture
    homes = build.selected.byte_homes
    assert [row[0] for row in homes] == [
        "base_pvsa1_incumbent_g74_both",
        "g89_class_complete_y1_program",
        "g88_conditional_y0_given_combined_y1",
    ]
    assert len({row[2] for row in homes}) == 3
    # Header + three sections + CRC is the entire counted member.
    assert sum(row[1] for row in homes) < len(build.selected.member_bytes)
    assert all(len(row[2]) == 64 for row in homes)


def test_strict_parse_reencode_crc_and_member_sha(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    _base, build, _receiver, _result = exact_fixture
    member = build.selected.member_bytes
    parsed = parse_sequential_typed_product(
        member,
        expected_member_sha256=hashlib.sha256(member).hexdigest(),
    )
    assert type(parsed) is ParsedSequentialTypedProductV1
    assert (
        encode_sequential_typed_product(
            base_pvsa_member_bytes=parsed.base_pvsa_member_bytes,
            g89_program_bytes=parsed.g89_program_bytes,
            g88_conditional_operand_bytes=parsed.g88_conditional_operand_bytes,
        )
        == member
    )
    tampered = member[:-1] + bytes((member[-1] ^ 1,))
    with pytest.raises(SequentialTypedProductError, match="CRC32"):
        parse_sequential_typed_product(tampered)


def test_incumbent_g89_boundary_collision_fails_closed() -> None:
    base = _current_base_member()
    conditional = _conditional(base)
    member = encode_sequential_typed_product(
        base_pvsa_member_bytes=base,
        g89_program_bytes=_program(collide_with_incumbent=True).to_bytes(),
        g88_conditional_operand_bytes=conditional.to_bytes(),
    )
    with pytest.raises(SequentialTypedProductError, match="donor collision"):
        parse_sequential_typed_product(member)


def test_g88_foreign_key_must_bind_exact_base() -> None:
    base = _current_base_member()
    wrong = replace(
        _conditional(base),
        base_pvsa_member_sha256="0" * 64,
    )
    member = encode_sequential_typed_product(
        base_pvsa_member_bytes=base,
        g89_program_bytes=_program().to_bytes(),
        g88_conditional_operand_bytes=wrong.to_bytes(),
    )
    with pytest.raises(SequentialTypedProductError, match="foreign keys"):
        parse_sequential_typed_product(member)


def test_streaming_refuses_more_than_batch16(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    _base, _build, receiver, _result = exact_fixture
    with pytest.raises(SequentialTypedProductError, match=r"1\.\.16"):
        receiver.render_camera_pair_batch(tuple(range(17)))


def test_g83_metadata_is_exact_state_shaped_but_not_admitted(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    _base, build, _receiver, result = exact_fixture
    metadata = build_g83_state_metadata(build=build, bounded_proof=result)
    value = metadata.to_dict()
    assert metadata.receiver_transition_chain[-1] == CAUSAL_TRANSITION_ID
    assert value["archive"]["sha256"] == build.outer_build.selected.archive_sha256
    assert value["allocator_schema_ready"] is True
    assert value["g83_admission_ready"] is False
    assert value["complete_component_row"] is None
    assert value["conditioning_state_sha256"] == build.conditioning_state_sha256
    assert value["conditional_fit_authority"] == CONDITIONAL_FIT_AUTHORITY
    assert value["g95_fit_receipt_sha256"] is None
    assert tuple(value["blockers"]) == OPEN_BLOCKERS
    assert value["candidate_claim"] is False
    assert value["score_claim"] is False


def test_g83_metadata_refuses_bounded_proof_from_another_product_state(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    base, first_build, _receiver, first_result = exact_fixture
    second_build = build_sequential_typed_archive(
        base_pvsa_member_bytes=base,
        g89_program_bytes=_program(lane_width_bias_q8=65).to_bytes(),
        g88_conditional_operand_bytes=_conditional(base).to_bytes(),
    )
    second_result = second_build.selected.open_receiver(verify_member_effects=False).decode_pair(0)

    assert first_result.conditioning_state_sha256 != (second_result.conditioning_state_sha256)
    with pytest.raises(SequentialTypedProductError, match="different exact product"):
        build_g83_state_metadata(
            build=first_build,
            bounded_proof=second_result,
        )


def test_batch_result_refuses_forged_array_receipt(
    exact_fixture: tuple[
        bytes,
        SequentialTypedArchiveBuildV1,
        SequentialTypedProductReceiverV1,
        SequentialTypedBatchResultV1,
    ],
) -> None:
    _base, _build, _receiver, result = exact_fixture
    with pytest.raises(SequentialTypedProductError, match="receipt SHA"):
        replace(result, camera_sha256="0" * 64)


def test_conditioning_state_hash_changes_with_g89_program() -> None:
    base = _current_base_member()
    conditional = _conditional(base)
    first = parse_sequential_typed_product(
        encode_sequential_typed_product(
            base_pvsa_member_bytes=base,
            g89_program_bytes=_program(lane_width_bias_q8=64).to_bytes(),
            g88_conditional_operand_bytes=conditional.to_bytes(),
        )
    )
    second = parse_sequential_typed_product(
        encode_sequential_typed_product(
            base_pvsa_member_bytes=base,
            g89_program_bytes=_program(lane_width_bias_q8=65).to_bytes(),
            g88_conditional_operand_bytes=conditional.to_bytes(),
        )
    )
    assert first.base_pvsa.member_sha256 == second.base_pvsa.member_sha256
    assert first.g89_program.sha256 != second.g89_program.sha256
    assert first.conditioning_state_sha256 != second.conditioning_state_sha256
