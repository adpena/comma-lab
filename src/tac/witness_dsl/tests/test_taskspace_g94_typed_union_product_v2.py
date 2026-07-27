# SPDX-License-Identifier: MIT
"""Contract proofs for the G94 V2 sparse-Y1/exclusive-Y0 sum product."""

from __future__ import annotations

import ast
import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g88_population_conditional_y0_pvsa_v1 import (
    ConditionalY0ControlV1,
    PopulationConditionalOperandV1,
)
from tac.witness_dsl.taskspace_g94_typed_union_product_v2 import (
    FINAL_Y1_G95_REFIT_BLOCKER,
    PUBLIC_INFLATE_BLOCKER,
    UPSTREAM_N600_BLOCKER,
    ConditionalY0OwnerTagV2,
    G88ConditionalY0OwnerV2,
    G94V2TypedUnionProductError,
    G94V2TypedUnionReceiver,
    G95PopulationChartY0OwnerV2,
    build_g94_v2_outer_archive,
    encode_g94_v2_preconditional_product,
    encode_g94_v2_typed_union_product,
    parse_g94_v2_preconditional_product,
    parse_g94_v2_typed_union_product,
)
from tac.witness_dsl.taskspace_g95_population_pose_preimage_chart_v1 import (
    encode_population_pose_preimage_basis,
    encode_population_pose_preimage_coefficient_chunk,
    parse_population_pose_preimage_basis,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)
from tac.witness_dsl.taskspace_sparse_atlas_cumulative_lowering_v1 import (
    SparseAtlasCumulativeReceiverV1,
    SparseAtlasY1OperandV1,
    SparseAtlasY1StepV1,
)

CURRENT_BASE_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/g85_pvsa_public_receiver_20260727_r1/archive.zip")
CURRENT_BASE_ARCHIVE_SHA256 = "b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd"
CURRENT_BASE_MEMBER_SHA256 = "d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31"
SEMANTIC_P_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def preconditional_case():
    if not CURRENT_BASE_ARCHIVE.is_file():
        pytest.skip("source-backed current G85 base archive is absent")
    archive = CURRENT_BASE_ARCHIVE.read_bytes()
    assert _sha256(archive) == CURRENT_BASE_ARCHIVE_SHA256
    outer = parse_taskspace_outer_archive(
        archive,
        expected_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
    )
    assert outer.member_sha256 == CURRENT_BASE_MEMBER_SHA256
    step = SparseAtlasY1StepV1(
        operand_id="g72:0000_0001:Road:d0:a1:p0",
        operand=RoleAwareBoundaryShearletOperandV1(
            frame_selector=SelectedPreimageFrameSelectorV1.Y1,
            atoms=(
                BoundaryShearletAtomV1(
                    pair_index=0,
                    role="Road",
                    center_y=240,
                    center_x=494,
                    scale_y=4,
                    scale_x=8,
                    shear_q4=0,
                    amplitude_q4=64,
                ),
            ),
        ),
    )
    sparse = SparseAtlasY1OperandV1(
        semantic_p_sha256=SEMANTIC_P_SHA256,
        base_archive_sha256=CURRENT_BASE_ARCHIVE_SHA256,
        base_pvsa_member_sha256=outer.member_sha256,
        g90_aggregate_sha256="1" * 64,
        g90_aggregate_self_sha256="2" * 64,
        g92_plan_sha256="3" * 64,
        steps=(step,),
    )
    member = encode_g94_v2_preconditional_product(
        base_pvsa_member_bytes=outer.member_bytes,
        sparse_y1_operand_bytes=sparse.to_bytes(),
    )
    parsed = parse_g94_v2_preconditional_product(
        member,
        expected_member_sha256=_sha256(member),
    )
    receiver = SparseAtlasCumulativeReceiverV1.open(
        base_pvsa_member_bytes=parsed.base_pvsa_member_bytes,
        sparse_operand_bytes=parsed.sparse_y1_operand_bytes,
        expected_sparse_operand_sha256=parsed.sparse_y1_operand.sha256,
        verify_member_effects=False,
    )
    return parsed, receiver


def _g88_owner(preconditional) -> G88ConditionalY0OwnerV2:
    operand = PopulationConditionalOperandV1(
        base_pvsa_member_sha256=preconditional.base_pvsa.member_sha256,
        semantic_p_sha256=preconditional.base_pvsa.semantic_p_sha256,
        controls=(ConditionalY0ControlV1.copy_conditional_y1(0),),
    )
    return G88ConditionalY0OwnerV2.parse(operand.to_bytes())


def _g95_owner(
    preconditional,
    sparse_receiver: SparseAtlasCumulativeReceiverV1,
    *,
    parent_sha256: str | None = None,
) -> G95PopulationChartY0OwnerV2:
    first_chunk_ids = tuple(range(16))
    first_chunk = sparse_receiver.render_final_preconditional_batch(first_chunk_ids).preconditional_camera_pairs
    basis_bytes = encode_population_pose_preimage_basis(
        g94_product_member_sha256=parent_sha256 or preconditional.member_sha256,
        g94_conditioning_state_sha256=preconditional.conditioning_state_sha256,
        # This structural fixture is deliberately not a fresh n600 refit.
        whole_preconditional_camera_sha256="4" * 64,
        selected_target_table_sha256="5" * 64,
        posenet_weights_sha256="6" * 64,
        basis_q=np.asarray([[[[1, 0, 0]]]], dtype=np.int8),
        basis_scales=np.ones(1, dtype=np.float32),
    )
    basis = parse_population_pose_preimage_basis(basis_bytes)
    chunk_bytes: list[bytes] = []
    for start in range(0, 600, 16):
        pair_ids = tuple(range(start, min(start + 16, 600)))
        coefficients = np.zeros((len(pair_ids), basis.rank), dtype=np.int16)
        preconditional_sha = "7" * 64
        if start == 0:
            coefficients[0, 0] = 1
            preconditional_sha = _sha256(memoryview(first_chunk).cast("B"))
        chunk_bytes.append(
            encode_population_pose_preimage_coefficient_chunk(
                basis_object_sha256=basis.object_sha256,
                population_state_key_sha256=basis.population_state_key,
                preconditional_camera_sha256=preconditional_sha,
                selected_target_sha256=hashlib.sha256(f"fixture-target-{start}".encode()).hexdigest(),
                source_pair_ids=pair_ids,
                rank=basis.rank,
                coefficients_q=coefficients,
                coefficient_scales=np.ones(basis.rank, dtype=np.float32),
            )
        )
    return G95PopulationChartY0OwnerV2.parse(
        basis_bytes=basis_bytes,
        chunk_bytes=tuple(chunk_bytes),
    )


def test_g88_branch_counts_each_object_once_and_preserves_final_g98_y1(
    preconditional_case,
) -> None:
    preconditional, _sparse_receiver = preconditional_case
    owner = _g88_owner(preconditional)
    build = build_g94_v2_outer_archive(
        preconditional=preconditional,
        y0_owner=owner,
    )

    assert build.stored == build.deflated == build.selected
    assert build.outer_build.stored.encoding is OuterArchiveEncoding.STORED
    assert build.outer_build.deflated.encoding is OuterArchiveEncoding.DEFLATED
    assert build.selected.y0_owner.tag is ConditionalY0OwnerTagV2.G88_POPULATION_CONDITIONAL
    assert build.selected.member_bytes.count(preconditional.base_pvsa_member_bytes) == 1
    assert build.selected.member_bytes.count(preconditional.sparse_y1_operand_bytes) == 1
    assert build.selected.member_bytes.count(owner.operand_bytes) == 1
    assert len({sha for _name, _count, sha in build.selected.byte_homes}) == len(build.selected.byte_homes)
    assert PUBLIC_INFLATE_BLOCKER in build.selected.open_blockers

    receiver = G94V2TypedUnionReceiver.open(
        build.selected,
        verify_member_effects=False,
    )
    first = receiver.decode_pair(0)
    second = receiver.decode_pair(0)
    assert first.camera_sha256 == second.camera_sha256
    assert first.final_y1_sha256 == second.final_y1_sha256
    assert np.array_equal(first.camera_pairs[:, 1], first.preconditional_camera_pairs[:, 1])
    assert first.changed_y0_values > 0
    assert first.score_claim is False


@pytest.mark.timeout(300)
def test_g95_branch_is_p_once_complete_indexed_and_y1_identity_safe(
    preconditional_case,
) -> None:
    preconditional, sparse_receiver = preconditional_case
    owner = _g95_owner(preconditional, sparse_receiver)
    build = build_g94_v2_outer_archive(
        preconditional=preconditional,
        y0_owner=owner,
    )

    assert build.selected.y0_owner.tag is ConditionalY0OwnerTagV2.G95_POPULATION_CHART
    assert len(owner.chunks) == 38
    assert build.selected.member_bytes.count(owner.basis_bytes) == 1
    assert all(build.selected.member_bytes.count(raw) == 1 for raw in owner.chunk_bytes)
    assert FINAL_Y1_G95_REFIT_BLOCKER in build.selected.open_blockers
    assert tuple(pair_id for chunk in owner.chunks for pair_id in chunk.source_pair_ids) == tuple(range(600))

    receiver = G94V2TypedUnionReceiver.open(
        build.selected,
        verify_member_effects=False,
    )
    result = receiver.decode_pair(0)
    assert result.changed_y0_values > 0
    assert np.array_equal(result.camera_pairs[:, 1], result.preconditional_camera_pairs[:, 1])
    # The fixture cannot manufacture the owed whole-state refit.  The real
    # population proof therefore fails closed at the first foreign chunk/hash.
    with pytest.raises(G94V2TypedUnionProductError, match=r"chunk|whole-state"):
        receiver.verify_g95_whole_state()


def test_closed_union_refuses_wrong_parent_and_trailing_second_owner_bytes(
    preconditional_case,
) -> None:
    preconditional, sparse_receiver = preconditional_case
    with pytest.raises(G94V2TypedUnionProductError, match="exactly one closed Y0 owner"):
        encode_g94_v2_typed_union_product(
            preconditional=preconditional,
            y0_owner=(_g88_owner(preconditional),),  # type: ignore[arg-type]
        )

    foreign = _g95_owner(
        preconditional,
        sparse_receiver,
        parent_sha256="8" * 64,
    )
    with pytest.raises(G94V2TypedUnionProductError, match="another preconditional"):
        parse_g94_v2_typed_union_product(
            encode_g94_v2_typed_union_product(
                preconditional=preconditional,
                y0_owner=foreign,
            )
        )

    g88 = _g88_owner(preconditional)
    valid = encode_g94_v2_typed_union_product(
        preconditional=preconditional,
        y0_owner=g88,
    )
    injected_prefix = valid[:-4] + foreign.basis_bytes
    injected = injected_prefix + struct.pack(
        ">I",
        zlib.crc32(injected_prefix) & 0xFFFFFFFF,
    )
    with pytest.raises(G94V2TypedUnionProductError, match="trailing or double-owner"):
        parse_g94_v2_typed_union_product(injected)


def test_receiver_module_is_scorer_free_and_truth_labels_remain_negative() -> None:
    module_path = Path(__file__).parents[1] / "taskspace_g94_typed_union_product_v2.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names} | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        forbidden in imported.lower()
        for imported in imports
        for forbidden in ("torch", "posenet", "segnet", "evaluate")
    )
    assert "candidate_claim: Literal[False] = False" in source
    assert "score_claim: Literal[False] = False" in source
    assert UPSTREAM_N600_BLOCKER in source
