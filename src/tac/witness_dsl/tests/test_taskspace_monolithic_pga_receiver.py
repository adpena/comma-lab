# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryCoefficientDelta,
    ReceiverRealizationProfileV1,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator
from tac.witness_dsl import ep725_levelset_predictor_adapter as ep725
from tac.witness_dsl import taskspace_monolithic_pga_receiver as pga
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    Frame1AnchoredY0FibreControlV1,
)
from tac.witness_dsl.ep725_levelset_predictor_adapter import (
    EP725_SOURCE_DIRECTORY,
    Ep725CountedMemberCausalSurfaceV3,
    Ep725EphemeralRuntimeSurfaceV2,
    Ep725LevelsetPredictorAdapterError,
    decode_ep725_counted_member_ephemeral_surface,
    inspect_ep725_source,
)
from tac.witness_dsl.generative_taskspace_correction import (
    EncoderOnlyTeacherEvidenceV1,
    GenerativeCorrectionProgramV1,
)
from tac.witness_dsl.predictor_preserving_coupled_preimage import (
    CorrectedY1SupportCopyCellV1,
    PredictorCameraPairSurfaceV1,
    PredictorPreservingA3Mode,
    PredictorPreservingA3ProgramV1,
    SparseConstantRGBCellV1,
    compile_predictor_preserving_a3,
)
from tac.witness_dsl.predictor_preserving_taskspace_overlay import (
    overlay_g_on_predictor_camera_y1,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    OuterArchiveEncoding,
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pass_conditional_a import (
    DecodedPassConditionalAV1,
    compile_pass_conditional_a,
)
from tac.witness_dsl.taskspace_pass_semantic_g import (
    DecodedPassSemanticGEnvelopeV1,
    compile_pass_semantic_g_envelope,
)
from tac.witness_dsl.taskspace_post_g8_conditional_a import (
    compile_post_g8_conditional_a,
)
from tac.witness_dsl.taskspace_predictor_v2_consumer_seam import (
    apply_generative_taskspace_correction_v2,
    compile_coupled_preimage_program_v2,
    compile_generative_taskspace_correction_v2,
)
from tac.witness_dsl.taskspace_same_class_realization_repair import (
    DecodedSameClassRealizationRepairPGAIntegrationV1,
    EncoderOnlyExactTargetLabelCustodyV1,
    SameClassRealizationRepairProgramV1,
    SameClassRealizationRepairRunV1,
    SameClassSemanticRoleV1,
    compile_same_class_realization_repair_for_pga_sections,
    decode_same_class_realization_repair_from_pga_sections,
)


def _sha256(payload: bytes | memoryview) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _structural_packets() -> tuple[bytes, bytes, bytes]:
    return b"LVLS1\x00structural-P", b"TACG1C\x00\x00structural-G", b'{"structural":"A"}'


def _nested_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("nested.bin", b"forbidden")
    return output.getvalue()


# TIMEOUT BUDGET 2026-09-03 (ddm_ql1). These tests drive the ep725 COLD decode, which runs four
# full decodes (canonical NumPy x2 then shipped-runtime subprocess x2, adapter
# _decode_executable_source_materialized). MEASURED: one shipped n2 decode costs 15.6-16.7 s and the
# whole cold path 65.6 s on an idle machine -- over the repo-wide pytest-timeout ceiling of 60 s
# (pyproject.toml). The cold surface is a module-scoped cache, so ANY test in this module can be the
# one that pays for filling it; the budget therefore belongs at module scope, not on one test.
# 180 s is not a new number: it is what this module's own decode call already asks for
# (timeout_seconds=180.0) and what the single pre-existing @pytest.mark.timeout(180) in this file
# already used. This raises a harness watchdog only -- every assertion still runs unchanged.
pytestmark = pytest.mark.timeout(180)


def test_strict_member_roundtrip_has_exact_p_g_a_and_optional_t_directory() -> None:
    predictor, correction, preimage = _structural_packets()
    for terminal in (None, b"terminal-quotient"):
        member = pga.encode_taskspace_monolithic_pga_member(
            predictor,
            correction,
            preimage,
            terminal_quotient_packet=terminal,
        )
        parsed = pga.parse_taskspace_monolithic_pga_member(member)
        expected_roles = pga.REQUIRED_ROLE_ORDER if terminal is None else pga.OPTIONAL_ROLE_ORDER

        assert tuple(row.role for row in parsed.sections) == expected_roles
        assert tuple(row.payload for row in parsed.sections[:3]) == (predictor, correction, preimage)
        assert (
            parsed.sections[0].offset == pga._MEMBER_PREFIX.size + len(parsed.sections) * pga._SECTION_DESCRIPTOR.size
        )
        assert all(
            left.stop == right.offset for left, right in zip(parsed.sections[:-1], parsed.sections[1:], strict=True)
        )
        assert parsed.sections[-1].stop == len(member)
        assert parsed.member_sha256 == _sha256(member)
        assert parsed.descriptor_manifest_sha256 == _sha256(
            pga._canonical_json([row.custody_dict() for row in parsed.sections])
        )


def test_outer_codec_is_the_only_zip_builder_and_reopens_opaque_exact_member() -> None:
    predictor, correction, preimage = _structural_packets()
    built = pga.build_taskspace_monolithic_pga_archive(predictor, correction, preimage)
    reopened = parse_taskspace_outer_archive(
        built.selected.archive_bytes,
        expected_encoding=built.selected.encoding,
        expected_archive_sha256=built.selected.archive_sha256,
        expected_member_sha256=built.selected.member_sha256,
    )
    parsed = pga.parse_taskspace_monolithic_pga_member(reopened.member_bytes)

    assert built.selected.encoding in {OuterArchiveEncoding.STORED, OuterArchiveEncoding.DEFLATED}
    assert reopened.member_bytes == parsed.member_bytes
    assert reopened.member_name == "0.bin"
    assert not zipfile.is_zipfile(io.BytesIO(reopened.member_bytes))


@pytest.mark.parametrize("role_index", range(3), ids=("predictor", "correction", "preimage"))
def test_directory_hash_and_crc_reject_each_corrupted_exact_section(role_index: int) -> None:
    member = bytearray(pga.encode_taskspace_monolithic_pga_member(*_structural_packets()))
    parsed = pga.parse_taskspace_monolithic_pga_member(bytes(member))
    section = parsed.sections[role_index]
    member[section.offset + section.byte_length // 2] ^= 0x01

    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="SHA-256 mismatch"):
        pga.parse_taskspace_monolithic_pga_member(bytes(member))


@pytest.mark.parametrize(
    "mutation",
    ("role-order", "descriptor-reserved", "offset", "length", "crc", "truncated", "trailing"),
)
def test_directory_order_range_crc_and_exact_eof_fail_closed(mutation: str) -> None:
    member = bytearray(pga.encode_taskspace_monolithic_pga_member(*_structural_packets()))
    first_descriptor = pga._MEMBER_PREFIX.size
    if mutation == "role-order":
        member[first_descriptor] = pga._ROLE_CODE[pga.TaskspaceMonolithicPGARole.GENERATIVE_CORRECTION]
    elif mutation == "descriptor-reserved":
        member[first_descriptor + 1] = 1
    elif mutation == "offset":
        struct.pack_into(">Q", member, first_descriptor + 4, len(member) - 1)
    elif mutation == "length":
        struct.pack_into(">Q", member, first_descriptor + 12, len(member))
    elif mutation == "crc":
        member[first_descriptor + 20] ^= 0x01
    elif mutation == "truncated":
        member = member[:-1]
    elif mutation == "trailing":
        member.extend(b"unconsumed")
    else:  # pragma: no cover - closed parametrization
        raise AssertionError(mutation)

    with pytest.raises(pga.TaskspaceMonolithicPGAError):
        pga.parse_taskspace_monolithic_pga_member(bytes(member))


@pytest.mark.parametrize("section_index", range(3), ids=("predictor", "correction", "preimage"))
def test_nested_zip_is_refused_in_every_counted_role(section_index: int) -> None:
    packets = list(_structural_packets())
    packets[section_index] = _nested_zip()
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="nested ZIP"):
        pga.encode_taskspace_monolithic_pga_member(*packets)


def test_empty_or_partial_member_construction_is_impossible() -> None:
    predictor, correction, preimage = _structural_packets()
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="nonempty immutable bytes"):
        pga.encode_taskspace_monolithic_pga_member(b"", correction, preimage)
    with pytest.raises(pga.TaskspaceMonolithicPGAError):
        pga.parse_taskspace_monolithic_pga_member(b"")
    prefix_only = pga._MEMBER_PREFIX.pack(pga.MEMBER_MAGIC, pga.MEMBER_VERSION, 2, 0)
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact P/G/A"):
        pga.parse_taskspace_monolithic_pga_member(prefix_only)


def _profile() -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (20, 80, 20),
            (240, 220, 40),
            (30, 30, 30),
            (220, 40, 40),
            (40, 80, 220),
        )
    )


def _teacher(surface: Ep725EphemeralRuntimeSurfaceV2) -> EncoderOnlyTeacherEvidenceV1:
    target = surface.predictor_state.labels.copy()
    target[0, 0, 0] = np.uint8((int(target[0, 0, 0]) + 1) % 5)
    target = np.ascontiguousarray(target)
    return EncoderOnlyTeacherEvidenceV1(
        pbr1_sha256="1" * 64,
        pbr2_sha256="2" * 64,
        target_labels_sha256=_sha256(memoryview(target).cast("B")),
        obligation_ir_sha256="4" * 64,
        oracle_evidence_sha256="5" * 64,
        dense_y_sha256="6" * 64,
        target_labels=target,
        teacher_event_count=1,
    )


def _g_packet(surface: Ep725EphemeralRuntimeSurfaceV2, *, mode_index: int = 0) -> bytes:
    state = surface.predictor_state
    program = GenerativeCorrectionProgramV1(
        boundary_coefficients=(
            BoundaryCoefficientDelta(
                state.source_pair_ids[0],
                "Road",
                mode_index,
                3.0,
            ),
        ),
        realization_profile=_profile(),
    )
    compiled = compile_generative_taskspace_correction_v2(
        state,
        program,
        teacher_evidence=_teacher(surface),
    )
    assert compiled.receipt.changed_cells > 0
    return compiled.packet


def _a_packet(
    surface: Ep725EphemeralRuntimeSurfaceV2,
    g_packet: bytes,
    *,
    variant: int = 0,
) -> bytes:
    state = surface.predictor_state
    decoded_g = apply_generative_taskspace_correction_v2(g_packet, predictor_state=state)
    controls = tuple(
        Frame1AnchoredY0FibreControlV1(
            pair_id,
            1,
            -2,
            (3 + variant, -4, 5),
        )
        for pair_id in state.source_pair_ids
    )
    return compile_coupled_preimage_program_v2(
        state,
        decoded_g,
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    ).packet


@cache
def _real_source_bytes() -> tuple[bytes, bytes]:
    if not Path(EP725_SOURCE_DIRECTORY).is_dir():
        pytest.skip("exact frozen ep725 SSD source is unavailable")
    source = inspect_ep725_source()
    return source.member, source.runtime


@cache
def _real_causal_surface_n2() -> Ep725CountedMemberCausalSurfaceV3:
    member, runtime = _real_source_bytes()
    return decode_ep725_counted_member_ephemeral_surface(
        member,
        shipped_runtime=runtime,
        pair_count=2,
        timeout_seconds=180.0,
    )


@cache
def _real_surface_n2() -> Ep725EphemeralRuntimeSurfaceV2:
    return _real_causal_surface_n2().ephemeral_surface


@dataclass(frozen=True)
class _RealFixture:
    surface: Ep725EphemeralRuntimeSurfaceV2
    causal_surface: Ep725CountedMemberCausalSurfaceV3
    runtime: bytes
    g_packet: bytes
    a_packet: bytes
    archive: bytes


@cache
def _real_fixture_n2() -> _RealFixture:
    causal_surface = _real_causal_surface_n2()
    surface = causal_surface.ephemeral_surface
    _member, runtime = _real_source_bytes()
    g_packet = _g_packet(surface)
    a_packet = _a_packet(surface, g_packet)
    built = pga.build_taskspace_monolithic_pga_archive(
        surface.predictor_state.predictor_program,
        g_packet,
        a_packet,
    )
    return _RealFixture(surface, causal_surface, runtime, g_packet, a_packet, built.selected.archive_bytes)


@dataclass(frozen=True)
class _RealA3Fixture:
    surface: Ep725EphemeralRuntimeSurfaceV2
    causal_surface: Ep725CountedMemberCausalSurfaceV3
    runtime: bytes
    g_packet: bytes
    pass_packet: bytes
    sparse_packet: bytes
    copy_packet: bytes
    pass_archive: bytes
    sparse_archive: bytes
    copy_archive: bytes


@cache
def _real_a3_fixture_n2() -> _RealA3Fixture:
    legacy = _real_fixture_n2()
    surface = legacy.surface
    state = surface.predictor_state
    decoded_g = apply_generative_taskspace_correction_v2(legacy.g_packet, predictor_state=state)
    overlay = overlay_g_on_predictor_camera_y1(surface.frame1_camera, state.labels, decoded_g)
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)

    pass_packet = compile_predictor_preserving_a3(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=overlay,
    ).packet

    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    sparse_row = operator.row_supports[0].indices[0]
    sparse_col = operator.col_supports[0].indices[0]
    base_rgb = surface.chronological_camera_frames[0, 0, sparse_row, sparse_col]
    changed_rgb = (
        (int(base_rgb[0]) + 1) % 256,
        (int(base_rgb[1]) + 1) % 256,
        (int(base_rgb[2]) + 1) % 256,
    )
    sparse_packet = compile_predictor_preserving_a3(
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1,
            constant_rgb_cells=(SparseConstantRGBCellV1(state.source_pair_ids[0], 0, 0, changed_rgb),),
        ),
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=overlay,
    ).packet

    copy_cell: CorrectedY1SupportCopyCellV1 | None = None
    for pair_index, scorer_row, scorer_col in np.argwhere(decoded_g.labels != state.labels):
        rows = operator.row_supports[int(scorer_row)].indices
        cols = operator.col_supports[int(scorer_col)].indices
        index = np.ix_(rows, cols, range(3))
        if not np.array_equal(
            predictor_surface.camera_p0[int(pair_index)][index],
            overlay.camera_y1[int(pair_index)][index],
        ):
            copy_cell = CorrectedY1SupportCopyCellV1(
                state.source_pair_ids[int(pair_index)],
                int(scorer_row),
                int(scorer_col),
            )
            break
    if copy_cell is None:  # pragma: no cover - exact source fixture invariant
        raise AssertionError("real ep725/G fixture has no non-no-op corrected-Y1 copy support")
    copy_packet = compile_predictor_preserving_a3(
        PredictorPreservingA3ProgramV1(
            PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1,
            corrected_y1_copy_cells=(copy_cell,),
        ),
        predictor_surface=predictor_surface,
        decoded_g=decoded_g,
        corrected_y1_overlay=overlay,
    ).packet

    def archive(packet: bytes) -> bytes:
        return pga.build_taskspace_monolithic_pga_archive(
            state.predictor_program,
            legacy.g_packet,
            packet,
        ).selected.archive_bytes

    return _RealA3Fixture(
        surface=surface,
        causal_surface=legacy.causal_surface,
        runtime=legacy.runtime,
        g_packet=legacy.g_packet,
        pass_packet=pass_packet,
        sparse_packet=sparse_packet,
        copy_packet=copy_packet,
        pass_archive=archive(pass_packet),
        sparse_archive=archive(sparse_packet),
        copy_archive=archive(copy_packet),
    )


@dataclass(frozen=True)
class _RealG8Fixture:
    surface: Ep725EphemeralRuntimeSurfaceV2
    causal_surface: Ep725CountedMemberCausalSurfaceV3
    runtime: bytes
    semantic_g_packet: bytes
    composite_g_packet: bytes
    post_g8_a_packet: bytes
    integration: DecodedSameClassRealizationRepairPGAIntegrationV1
    repair_program: SameClassRealizationRepairProgramV1
    target_custody: EncoderOnlyExactTargetLabelCustodyV1
    archive: bytes


@cache
def _real_g8_fixture_n2(repair_rgb_delta: int = 127) -> _RealG8Fixture:
    legacy = _real_fixture_n2()
    surface = legacy.surface
    state = surface.predictor_state
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    decoded_g = apply_generative_taskspace_correction_v2(
        legacy.g_packet,
        predictor_state=state,
    )
    overlay = overlay_g_on_predictor_camera_y1(
        surface.frame1_camera,
        state.labels,
        decoded_g,
    )
    pair_index, scorer_row, scorer_col = (int(value) for value in np.argwhere(decoded_g.labels >= 0)[0])
    semantic_class = int(decoded_g.labels[pair_index, scorer_row, scorer_col])
    semantic_role = (
        SameClassSemanticRoleV1.ROAD,
        SameClassSemanticRoleV1.LANE,
        SameClassSemanticRoleV1.UNDRIVABLE_BOUNDARY,
        SameClassSemanticRoleV1.MOVABLE,
        SameClassSemanticRoleV1.MY_CAR,
    )[semantic_class]
    operator = DisjointResizeOperator.build(
        camera_h=874,
        camera_w=1164,
        scorer_h=384,
        scorer_w=512,
    )
    camera_row = int(operator.row_supports[scorer_row].indices[0])
    camera_col = int(operator.col_supports[scorer_col].indices[0])
    base_rgb = overlay.camera_y1[pair_index, camera_row, camera_col]
    repair_rgb = tuple((int(value) + repair_rgb_delta) % 256 for value in base_rgb)
    repair_program = SameClassRealizationRepairProgramV1(
        (
            SameClassRealizationRepairRunV1(
                source_pair_id=state.source_pair_ids[pair_index],
                scorer_row=scorer_row,
                scorer_col_start=scorer_col,
                scorer_col_stop=scorer_col + 1,
                semantic_class=semantic_class,
                semantic_role=semantic_role,
                rgb_u8=repair_rgb,  # type: ignore[arg-type]
            ),
        )
    )
    target_labels = np.ascontiguousarray(decoded_g.labels.copy())
    target_custody = EncoderOnlyExactTargetLabelCustodyV1(
        source_artifact_sha256="a" * 64,
        source_member_name="gt_n600.npz::lstars",
        source_member_sha256="b" * 64,
        source_pair_ids=state.source_pair_ids,
        target_labels_sha256=_sha256(memoryview(target_labels).cast("B")),
        target_labels=target_labels,
    )
    proposal = compile_same_class_realization_repair_for_pga_sections(
        repair_program,
        predictor_section_payload=state.predictor_program,
        semantic_g_section_payload=legacy.g_packet,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
        target_custody=target_custody,
    )
    integration = decode_same_class_realization_repair_from_pga_sections(
        proposal.replacement_g_section_payload,
        predictor_section_payload=state.predictor_program,
        predictor_surface=predictor_surface,
        predictor_codec_id="LVLS1.v1",
        semantic_g_codec_id="TACG1C.v2",
    )
    post_a = compile_post_g8_conditional_a(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=predictor_surface,
        g8_integration=integration,
    )
    archive = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        proposal.replacement_g_section_payload,
        post_a.packet,
    ).selected.archive_bytes
    return _RealG8Fixture(
        surface=surface,
        causal_surface=legacy.causal_surface,
        runtime=legacy.runtime,
        semantic_g_packet=legacy.g_packet,
        composite_g_packet=proposal.replacement_g_section_payload,
        post_g8_a_packet=post_a.packet,
        integration=integration,
        repair_program=repair_program,
        target_custody=target_custody,
        archive=archive,
    )


@dataclass(frozen=True)
class _RealPassFixture:
    surface: Ep725EphemeralRuntimeSurfaceV2
    causal_surface: Ep725CountedMemberCausalSurfaceV3
    runtime: bytes
    pass_g_packet: bytes
    decoded_pass_g: DecodedPassSemanticGEnvelopeV1
    conditional_a_packet: bytes
    decoded_conditional_a: DecodedPassConditionalAV1
    archive: bytes


@cache
def _real_pass_fixture_n2(with_g8: bool = False) -> _RealPassFixture:
    legacy = _real_fixture_n2()
    surface = legacy.surface
    state = surface.predictor_state
    predictor_surface = PredictorCameraPairSurfaceV1.from_ep725(surface)
    g8 = _real_g8_fixture_n2() if with_g8 else None
    compiled_g = compile_pass_semantic_g_envelope(
        predictor_section_payload=state.predictor_program,
        predictor_surface=predictor_surface,
        repair_program=None if g8 is None else g8.repair_program,
        target_custody=None if g8 is None else g8.target_custody,
    )
    compiled_a = compile_pass_conditional_a(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=predictor_surface,
        pass_g=compiled_g.decoded,
    )
    archive = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        compiled_g.envelope,
        compiled_a.packet,
    ).selected.archive_bytes
    return _RealPassFixture(
        surface=surface,
        causal_surface=legacy.causal_surface,
        runtime=legacy.runtime,
        pass_g_packet=compiled_g.envelope,
        decoded_pass_g=compiled_g.decoded,
        conditional_a_packet=compiled_a.packet,
        decoded_conditional_a=compiled_a.decoded,
        archive=archive,
    )


def _install_cached_causal_decode(
    monkeypatch: pytest.MonkeyPatch,
    causal_surface: Ep725CountedMemberCausalSurfaceV3,
    runtime: bytes,
) -> None:
    expected_member = causal_surface.predictor_state.predictor_program

    def decode(
        counted_member: bytes,
        *,
        shipped_runtime: bytes,
        pair_count: int,
        timeout_seconds: float,
    ) -> Ep725CountedMemberCausalSurfaceV3:
        del timeout_seconds
        if counted_member != expected_member:
            raise Ep725LevelsetPredictorAdapterError("counted member differs from cached causal fixture")
        if shipped_runtime != runtime:
            raise Ep725LevelsetPredictorAdapterError("runtime differs from cached causal fixture")
        if pair_count != 2:
            raise Ep725LevelsetPredictorAdapterError("pair count differs from cached causal fixture")
        return causal_surface

    monkeypatch.setattr(pga, "decode_ep725_counted_member_ephemeral_surface", decode)


def _audit_cached_real(
    monkeypatch: pytest.MonkeyPatch,
    archive: bytes | None = None,
) -> pga.TaskspaceMonolithicPGABlockerReceiptV1:
    fixture = _real_fixture_n2()
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    return pga.audit_ep725_taskspace_monolithic_pga_archive(
        fixture.archive if archive is None else archive,
        predictor_runtime=fixture.runtime,
        pair_count=2,
    )


def _receive_cached_a3(
    monkeypatch: pytest.MonkeyPatch,
    archive: bytes,
) -> pga.DecodedTaskspaceMonolithicPGAV1:
    fixture = _real_a3_fixture_n2()
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    return pga.receive_ep725_taskspace_monolithic_pga_archive(
        archive,
        predictor_runtime=fixture.runtime,
        pair_count=2,
    )


def test_real_ep725_n2_p_g_camera_closes_then_a_blocks_with_exact_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_fixture_n2()
    receipt = _audit_cached_real(monkeypatch)
    parsed_outer = parse_taskspace_outer_archive(fixture.archive)
    parsed_member = pga.parse_taskspace_monolithic_pga_member(parsed_outer.member_bytes)

    assert receipt.blocker_code is pga.TaskspaceMonolithicPGABlockerCode.A_SIGNAL_LOSS_FULL_REPAINT
    assert receipt.sections[0].payload_sha256 == fixture.surface.predictor_state.predictor_program_sha256
    assert receipt.sections[1].payload_sha256 == _sha256(fixture.g_packet)
    assert receipt.sections[2].payload_sha256 == _sha256(fixture.a_packet)
    assert receipt.section_descriptor_manifest_sha256 == parsed_member.descriptor_manifest_sha256
    assert receipt.predictor_double_decode_exact is True
    assert receipt.counted_p_is_actual_decode_input is True
    assert receipt.explicit_runtime_is_actual_decode_input is True
    assert receipt.source_archive_read_for_decode is False
    assert receipt.predictor_causal_decode_receipt.counted_member_sha256 == receipt.sections[0].payload_sha256
    assert receipt.predictor_causal_decode_receipt.source_directory_or_archive_read is False
    assert receipt.predictor_causal_decode_receipt_sha256 == receipt.predictor_causal_decode_receipt.receipt_sha256
    assert receipt.deterministic_preblock_double_replay is True
    assert receipt.p_ephemeral_y1_surface_available is True
    assert receipt.g_ownership_derived_from_exact_p_g is True
    assert receipt.p_g_ownership_scope == "decoded_g_labels_not_equal_predictor_internal_labels"
    assert receipt.p_g_label_inequality_overlay_r_abi_bound is True
    assert receipt.g_exact_owned_overlay_applied is True
    assert receipt.unchanged_p_y1_camera_values_byte_identical_verified is True
    assert receipt.unchanged_p_y1_scorer_numerators_identical_verified is True
    assert receipt.p_g_complete_bounded_camera_y1_bytes is True
    assert receipt.same_class_realization_repair_coordinate_present is False
    assert receipt.through_r_target_realization_debt_closed is False
    assert receipt.target_labels_consumed is False
    assert receipt.scorer_or_evaluator_closure is False
    assert receipt.p_g_base_camera_y1_sha256 != receipt.p_g_corrected_camera_y1_sha256
    assert len(receipt.p_g_corrected_camera_frame_sha256) == 2
    assert receipt.a_exact_y1_binding_matches_full_repaint is True
    assert receipt.a_conditioned_on_predictor_preserving_y1 is False
    assert receipt.exact_counted_a_materialized is False
    assert receipt.all_counted_p_g_a_semantically_consumed is False
    assert receipt.scorer_y0_sha256 is None
    assert receipt.scorer_y1_sha256 is None
    assert receipt.factor2_camera_frame_sha256_chronological == ()
    assert receipt.complete_bounded_chronological_output is False
    assert receipt.scorer_invoked is False
    assert receipt.exact_score_claim is False
    assert receipt.candidate_archive_eligible is False
    assert receipt.originality_claim is False
    assert receipt.promotion_eligible is False
    assert receipt.research_only is True

    reopened = pga.parse_taskspace_monolithic_pga_blocker_receipt(receipt.to_receipt_bytes())
    assert reopened == receipt
    assert reopened.receipt_sha256 == receipt.receipt_sha256
    print("G3_EP725_N2_PGA_BLOCKER_RECEIPT=" + receipt.to_receipt_bytes().decode("ascii"))


def test_public_full_pga_receive_raises_a_scoped_signal_blocker_with_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_fixture_n2()
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    with pytest.raises(pga.TaskspaceMonolithicPGAReceiverBlocker) as caught:
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            fixture.archive,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    assert caught.value.code is pga.TaskspaceMonolithicPGABlockerCode.A_SIGNAL_LOSS_FULL_REPAINT
    assert caught.value.receipt is not None
    assert caught.value.receipt.p_g_complete_bounded_camera_y1_bytes is True
    assert caught.value.receipt.complete_bounded_chronological_output is False


def test_substituted_p_g_a_and_corrupted_outer_bytes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_fixture_n2()
    surface = fixture.surface
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)

    changed_p = bytearray(surface.predictor_state.predictor_program)
    changed_p[-1] ^= 0x01
    p_archive = pga.build_taskspace_monolithic_pga_archive(
        bytes(changed_p),
        fixture.g_packet,
        fixture.a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact counted ep725 member/runtime decode failed"):
        pga.audit_ep725_taskspace_monolithic_pga_archive(
            p_archive,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    changed_g = _g_packet(surface, mode_index=1)
    g_archive = pga.build_taskspace_monolithic_pga_archive(
        surface.predictor_state.predictor_program,
        changed_g,
        fixture.a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="source binding mismatch"):
        pga.audit_ep725_taskspace_monolithic_pga_archive(
            g_archive,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    changed_a = _a_packet(surface, changed_g, variant=7)
    a_archive = pga.build_taskspace_monolithic_pga_archive(
        surface.predictor_state.predictor_program,
        fixture.g_packet,
        changed_a,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="source binding mismatch"):
        pga.audit_ep725_taskspace_monolithic_pga_archive(
            a_archive,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    corrupted = bytearray(fixture.archive)
    corrupted[len(corrupted) // 2] ^= 0x01
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="strict outer archive parse"):
        pga.audit_ep725_taskspace_monolithic_pga_archive(
            bytes(corrupted),
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )


def test_optional_t_is_bound_by_grammar_but_refused_before_semantic_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_fixture_n2()
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    with_t = pga.build_taskspace_monolithic_pga_archive(
        fixture.surface.predictor_state.predictor_program,
        fixture.g_packet,
        fixture.a_packet,
        terminal_quotient_packet=b"counted-but-unconsumed-T",
    ).selected.archive_bytes

    with pytest.raises(pga.TaskspaceMonolithicPGAReceiverBlocker) as caught:
        pga.audit_ep725_taskspace_monolithic_pga_archive(
            with_t,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )
    assert caught.value.code is pga.TaskspaceMonolithicPGABlockerCode.TERMINAL_QUOTIENT_CONSUMER_OWED
    assert caught.value.receipt is None


def test_blocker_receipt_rejects_truth_and_numeric_type_smuggling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _audit_cached_real(monkeypatch)
    body = json.loads(receipt.to_receipt_bytes())
    body["p_g_complete_bounded_camera_y1_bytes"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="truth labels"):
        pga.parse_taskspace_monolithic_pga_blocker_receipt(forged_truth)

    body = json.loads(receipt.to_receipt_bytes())
    body["g_owned_cells"] = float(body["g_owned_cells"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact integers"):
        pga.parse_taskspace_monolithic_pga_blocker_receipt(forged_number)

    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="byte-canonical JSON"):
        pga.parse_taskspace_monolithic_pga_blocker_receipt(receipt.to_receipt_bytes() + b"\n")


def test_a_only_variant_cannot_claim_y0_counterfactual_before_overlay_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_fixture_n2()
    baseline = _audit_cached_real(monkeypatch)
    changed_a = _a_packet(fixture.surface, fixture.g_packet, variant=9)
    changed_archive = pga.build_taskspace_monolithic_pga_archive(
        fixture.surface.predictor_state.predictor_program,
        fixture.g_packet,
        changed_a,
    ).selected.archive_bytes
    changed = _audit_cached_real(monkeypatch, changed_archive)

    assert baseline.sections[0] == changed.sections[0]
    assert baseline.sections[1] == changed.sections[1]
    assert baseline.sections[2] != changed.sections[2]
    assert baseline.p_g_corrected_camera_y1_sha256 == changed.p_g_corrected_camera_y1_sha256
    assert baseline.p_g_corrected_camera_frame_sha256 == changed.p_g_corrected_camera_frame_sha256
    assert baseline.a_source_binding_sha256 != changed.a_source_binding_sha256
    assert baseline.scorer_y0_sha256 is changed.scorer_y0_sha256 is None
    assert baseline.complete_bounded_chronological_output is False
    assert changed.complete_bounded_chronological_output is False


def test_real_ep725_n2_full_monolithic_a3_modes_emit_exact_factor2_hash_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_a3_fixture_n2()
    inputs = (
        (PredictorPreservingA3Mode.PASS_P0_V1, fixture.pass_packet, fixture.pass_archive),
        (PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1, fixture.sparse_packet, fixture.sparse_archive),
        (PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1, fixture.copy_packet, fixture.copy_archive),
    )
    receipts: dict[PredictorPreservingA3Mode, pga.TaskspaceMonolithicPGAReceiverReceiptV1] = {}
    outputs: dict[PredictorPreservingA3Mode, pga.DecodedTaskspaceMonolithicPGAV1] = {}
    for mode, packet, archive in inputs:
        decoded = _receive_cached_a3(monkeypatch, archive)
        receipt = decoded.receipt
        receipts[mode] = receipt
        outputs[mode] = decoded
        parsed_outer = parse_taskspace_outer_archive(archive)
        parsed_member = pga.parse_taskspace_monolithic_pga_member(parsed_outer.member_bytes)

        assert receipt.archive_bytes == len(archive)
        assert receipt.archive_sha256 == _sha256(archive)
        assert receipt.member_bytes == len(parsed_member.member_bytes)
        assert receipt.member_sha256 == _sha256(parsed_member.member_bytes)
        assert receipt.member_crc32 == parsed_outer.member_crc32
        assert receipt.sections[0].payload_sha256 == fixture.surface.predictor_state.predictor_program_sha256
        assert receipt.sections[1].payload_sha256 == _sha256(fixture.g_packet)
        assert receipt.sections[2].payload_sha256 == _sha256(packet)
        assert receipt.sections[2].byte_length == len(packet) == receipt.a_receipt.packet_bytes
        assert receipt.section_descriptor_manifest_sha256 == parsed_member.descriptor_manifest_sha256
        assert receipt.a_receipt.mode == mode.value
        assert receipt.a_receipt.packet_sha256 == _sha256(packet)
        assert receipt.a_receipt_sha256 == receipt.a_receipt.receipt_sha256
        assert receipt.source_pair_ids == (0, 1)
        assert len(receipt.camera_y0_frame_sha256) == 2
        assert len(receipt.camera_y1_frame_sha256) == 2
        assert len(receipt.factor2_camera_frame_sha256_chronological) == 4
        assert receipt.factor2_camera_frame_sha256_chronological[0::2] == receipt.camera_y0_frame_sha256
        assert receipt.factor2_camera_frame_sha256_chronological[1::2] == receipt.camera_y1_frame_sha256
        assert receipt.camera_y1_frame_sha256 == receipt.p_g_corrected_camera_frame_sha256
        assert decoded.camera_y0.dtype == decoded.camera_y1.dtype == np.uint8
        assert decoded.camera_y0.shape == decoded.camera_y1.shape == (2, 874, 1164, 3)
        assert decoded.chronological_camera_frames.shape == (2, 2, 874, 1164, 3)
        assert np.array_equal(decoded.chronological_camera_frames[:, 0], decoded.camera_y0)
        assert np.array_equal(decoded.chronological_camera_frames[:, 1], decoded.camera_y1)
        assert decoded.camera_y0.flags.writeable is False
        assert decoded.camera_y1.flags.writeable is False
        assert decoded.chronological_camera_frames.flags.writeable is False
        assert receipt.exact_counted_p_reopened_and_matched is True
        assert receipt.exact_counted_g_reapplied is True
        assert receipt.exact_counted_a_strictly_parsed_and_source_bound is True
        assert receipt.exact_counted_a_materialized is True
        assert receipt.all_counted_p_g_a_semantically_consumed is True
        assert receipt.section_descriptors_sha_crc_and_exact_eof_verified is True
        assert receipt.deterministic_receiver_double_replay is True
        assert receipt.counted_p_is_actual_decode_input is True
        assert receipt.explicit_runtime_is_actual_decode_input is True
        assert receipt.source_archive_read_for_decode is False
        assert receipt.predictor_causal_decode_receipt.counted_member_sha256 == receipt.sections[0].payload_sha256
        assert receipt.predictor_causal_decode_receipt.source_directory_or_archive_read is False
        assert receipt.predictor_causal_decode_receipt_sha256 == receipt.predictor_causal_decode_receipt.receipt_sha256
        assert receipt.a_conditioned_on_predictor_preserving_y1 is True
        assert receipt.corrected_y1_unchanged_by_a is True
        assert receipt.outside_a_owned_p0_camera_values_preserved is True
        assert receipt.outside_a_owned_p0_scorer_numerators_preserved is True
        assert receipt.complete_bounded_chronological_output is True
        assert receipt.dense_frames_persisted is False
        assert receipt.a3_camera_seam_control_only is True
        assert receipt.full_a3_se3_xi_basis_universe_implemented is False
        assert receipt.se3_xi_inverse_solve_implemented is False
        assert receipt.encoder_side_a_row_selection_solved is False
        assert receipt.same_class_realization_repair_coordinate_present is False
        assert receipt.through_r_target_realization_debt_closed is False
        assert receipt.target_labels_consumed is False
        assert receipt.scorer_or_evaluator_closure is False
        assert receipt.scorer_invoked is False
        assert receipt.n600_evidence_claim is False
        assert receipt.exact_score_claim is False
        assert receipt.candidate_archive_eligible is False
        assert receipt.originality_claim is False
        assert receipt.promotion_eligible is False
        assert receipt.standalone_runtime_closure is False
        assert receipt.complete_data_in_code_lineage_audit is False
        assert receipt.research_only is True
        assert pga.parse_taskspace_monolithic_pga_receiver_receipt(receipt.to_receipt_bytes()) == receipt

    passed = receipts[PredictorPreservingA3Mode.PASS_P0_V1]
    sparse = receipts[PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1]
    copied = receipts[PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1]
    assert passed.camera_y0_frame_sha256 == passed.predictor_frame0_camera_sha256
    assert sparse.camera_y0_frame_sha256 != passed.camera_y0_frame_sha256
    assert copied.camera_y0_frame_sha256 != passed.camera_y0_frame_sha256
    assert passed.camera_y1_frame_sha256 == sparse.camera_y1_frame_sha256 == copied.camera_y1_frame_sha256
    assert passed.sections[:2] == sparse.sections[:2] == copied.sections[:2]
    assert (
        len({passed.sections[2].payload_sha256, sparse.sections[2].payload_sha256, copied.sections[2].payload_sha256})
        == 3
    )
    assert len({passed.archive_sha256, sparse.archive_sha256, copied.archive_sha256}) == 3
    assert not np.array_equal(
        outputs[PredictorPreservingA3Mode.PASS_P0_V1].camera_y0,
        outputs[PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1].camera_y0,
    )
    assert not np.array_equal(
        outputs[PredictorPreservingA3Mode.PASS_P0_V1].camera_y0,
        outputs[PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1].camera_y0,
    )
    assert np.array_equal(
        outputs[PredictorPreservingA3Mode.PASS_P0_V1].camera_y1,
        outputs[PredictorPreservingA3Mode.SPARSE_CONSTANT_RGB_V1].camera_y1,
    )
    assert np.array_equal(
        outputs[PredictorPreservingA3Mode.PASS_P0_V1].camera_y1,
        outputs[PredictorPreservingA3Mode.COPY_CORRECTED_Y1_SUPPORT_V1].camera_y1,
    )
    print("G3_EP725_N2_PGA_A3_RECEIPT_SHA256=" + sparse.receipt_sha256)
    print("G3_EP725_N2_PGA_A3_RECEIPT=" + sparse.to_receipt_bytes().decode("ascii"))


@pytest.mark.skipif(not EP725_SOURCE_DIRECTORY.is_dir(), reason="frozen ep725 source SSD is unavailable")
@pytest.mark.timeout(180)
def test_real_ep725_n2_monolithic_decodes_from_member_runtime_when_source_helpers_explode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_a3_fixture_n2()

    def explode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("historical source helper must not run during causal decode")

    monkeypatch.setattr(ep725, "_read_source_directory", explode)
    monkeypatch.setattr(ep725, "inspect_ep725_source", explode)

    decoded = pga.receive_ep725_taskspace_monolithic_pga_archive(
        fixture.sparse_archive,
        predictor_runtime=fixture.runtime,
        pair_count=2,
        timeout_seconds=180.0,
    )
    causal = decoded.receipt.predictor_causal_decode_receipt
    assert causal.counted_member_sha256 == fixture.causal_surface.causal_receipt.counted_member_sha256
    assert causal.explicit_runtime_sha256 == fixture.causal_surface.causal_receipt.explicit_runtime_sha256
    assert (
        causal.chronological_raw_prefix_sha256 == fixture.causal_surface.causal_receipt.chronological_raw_prefix_sha256
    )
    assert causal.labels_sha256 == fixture.causal_surface.causal_receipt.labels_sha256
    assert causal.source_directory_or_archive_read is False
    assert causal.source_archive_bytes_read == 0
    assert decoded.receipt.source_archive_read_for_decode is False

    substituted_member = bytearray(fixture.surface.predictor_state.predictor_program)
    substituted_member[-1] ^= 0x01
    substituted_archive = pga.build_taskspace_monolithic_pga_archive(
        bytes(substituted_member),
        fixture.g_packet,
        fixture.sparse_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact counted ep725 member/runtime decode failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            substituted_archive,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    substituted_runtime = bytearray(fixture.runtime)
    substituted_runtime[-1] ^= 0x01
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact counted ep725 member/runtime decode failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            fixture.sparse_archive,
            predictor_runtime=bytes(substituted_runtime),
            pair_count=2,
        )


def test_real_ep725_n2_a3_deletion_corruption_and_foreign_source_substitution_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_a3_fixture_n2()
    surface = fixture.surface
    state = surface.predictor_state
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)

    changed_p = bytearray(state.predictor_program)
    changed_p[-1] ^= 0x01
    substituted_p = pga.build_taskspace_monolithic_pga_archive(
        bytes(changed_p),
        fixture.g_packet,
        fixture.sparse_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact counted ep725 member/runtime decode failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            substituted_p,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    truncated_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        fixture.g_packet,
        fixture.sparse_packet[:-1],
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="counted A3 failed exact decode"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            truncated_a,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    corrupted_packet = bytearray(fixture.sparse_packet)
    corrupted_packet[-1] ^= 0x01
    corrupted_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        fixture.g_packet,
        bytes(corrupted_packet),
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="counted A3 failed exact decode"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            corrupted_a,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    changed_g = _g_packet(surface, mode_index=1)
    substituted_g = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        changed_g,
        fixture.sparse_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="counted A3 failed exact decode"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            substituted_g,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )

    decoded_changed_g = apply_generative_taskspace_correction_v2(changed_g, predictor_state=state)
    changed_overlay = overlay_g_on_predictor_camera_y1(surface.frame1_camera, state.labels, decoded_changed_g)
    foreign_a = compile_predictor_preserving_a3(
        PredictorPreservingA3ProgramV1(PredictorPreservingA3Mode.PASS_P0_V1),
        predictor_surface=PredictorCameraPairSurfaceV1.from_ep725(surface),
        decoded_g=decoded_changed_g,
        corrected_y1_overlay=changed_overlay,
    ).packet
    substituted_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        fixture.g_packet,
        foreign_a,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="counted A3 failed exact decode"):
        pga.receive_ep725_taskspace_monolithic_pga_archive(
            substituted_a,
            predictor_runtime=fixture.runtime,
            pair_count=2,
        )


def test_full_receiver_receipt_rejects_truth_and_numeric_smuggling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_a3_fixture_n2()
    receipt = _receive_cached_a3(monkeypatch, fixture.sparse_archive).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["complete_bounded_chronological_output"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="truth labels"):
        pga.parse_taskspace_monolithic_pga_receiver_receipt(forged_truth)

    body = json.loads(receipt.to_receipt_bytes())
    body["archive_bytes"] = float(body["archive_bytes"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact integers"):
        pga.parse_taskspace_monolithic_pga_receiver_receipt(forged_number)


def test_composite_g8_then_post_g8_a_receiver_closes_nested_custody_and_cached_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _real_g8_fixture_n2()
    cached = pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        fixture.archive,
        causal_surface=fixture.causal_surface,
    )
    assert type(cached) is pga.DecodedTaskspaceMonolithicPGAG8V1
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    default = pga.receive_ep725_taskspace_monolithic_pga_archive(
        fixture.archive,
        predictor_runtime=fixture.runtime,
        pair_count=2,
    )
    assert type(default) is pga.DecodedTaskspaceMonolithicPGAG8V1
    assert cached.receipt == default.receipt
    assert np.array_equal(cached.chronological_camera_frames, default.chronological_camera_frames)

    receipt = cached.receipt
    assert receipt.sections[0].payload_sha256 == fixture.surface.predictor_state.predictor_program_sha256
    assert receipt.sections[1].payload_sha256 == _sha256(fixture.composite_g_packet)
    assert receipt.sections[2].payload_sha256 == _sha256(fixture.post_g8_a_packet)
    assert receipt.inner_semantic_g_packet_sha256 == _sha256(fixture.semantic_g_packet)
    assert receipt.inner_g8_repair_packet_sha256 == fixture.integration.decoded_repair.receipt.packet_sha256
    assert receipt.pre_g8_corrected_y1_sha256 == fixture.integration.surface.corrected_y1_camera_sha256
    assert receipt.post_g8_corrected_y1_sha256 == fixture.integration.decoded_repair.receipt.output_camera_y1_sha256
    assert receipt.pre_g8_corrected_y1_sha256 != receipt.post_g8_corrected_y1_sha256
    assert receipt.post_g8_a_receipt.post_g8_corrected_y1_sha256 == receipt.post_g8_corrected_y1_sha256
    assert receipt.camera_y1_frame_sha256 == receipt.factor2_camera_frame_sha256_chronological[1::2]
    assert np.array_equal(cached.camera_y1, fixture.integration.post_g8_corrected_y1)
    assert receipt.exact_composite_g_strictly_parsed is True
    assert receipt.semantic_overlay_applied_before_g8 is True
    assert receipt.exact_g8_repair_applied_after_semantic_g is True
    assert receipt.semantic_g_labels_unchanged_by_g8 is True
    assert receipt.exact_post_g8_a_strictly_parsed_and_source_bound is True
    assert receipt.post_g8_y1_unchanged_by_a is True
    assert receipt.cached_p_surface_requires_per_archive_identity_match is True
    assert receipt.same_class_realization_repair_coordinate_present is True
    assert receipt.through_r_target_realization_debt_closed is False
    assert receipt.scorer_invoked is False
    assert receipt.n600_evidence_claim is False
    assert receipt.exact_score_claim is False
    assert receipt.candidate_archive_eligible is False
    assert pga.parse_taskspace_monolithic_pga_g8_receiver_receipt(receipt.to_receipt_bytes()) == receipt


def test_cached_p_entrypoint_preserves_legacy_receipt_and_rechecks_archive_owned_p(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy = _real_a3_fixture_n2()
    cached = pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        legacy.sparse_archive,
        causal_surface=legacy.causal_surface,
    )
    assert type(cached) is pga.DecodedTaskspaceMonolithicPGAV1
    _install_cached_causal_decode(monkeypatch, legacy.causal_surface, legacy.runtime)
    default = pga.receive_ep725_taskspace_monolithic_pga_archive(
        legacy.sparse_archive,
        predictor_runtime=legacy.runtime,
        pair_count=2,
    )
    assert type(default) is pga.DecodedTaskspaceMonolithicPGAV1
    assert cached.receipt.to_receipt_bytes() == default.receipt.to_receipt_bytes()
    assert cached.receipt.receipt_sha256 == default.receipt.receipt_sha256

    changed_p = bytearray(legacy.surface.predictor_state.predictor_program)
    changed_p[-1] ^= 1
    substituted = pga.build_taskspace_monolithic_pga_archive(
        bytes(changed_p),
        legacy.g_packet,
        legacy.sparse_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="directory-owned P differs"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            substituted,
            causal_surface=legacy.causal_surface,
        )
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact causal surface type"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            legacy.sparse_archive,
            causal_surface=object(),  # type: ignore[arg-type]
        )


def test_composite_and_a_packet_domains_cannot_be_crossed() -> None:
    g8 = _real_g8_fixture_n2()
    legacy = _real_a3_fixture_n2()
    composite_with_legacy_a = pga.build_taskspace_monolithic_pga_archive(
        g8.surface.predictor_state.predictor_program,
        g8.composite_g_packet,
        legacy.pass_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="legacy A3 is bound before G8"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            composite_with_legacy_a,
            causal_surface=g8.causal_surface,
        )

    bare_g_with_post_a = pga.build_taskspace_monolithic_pga_archive(
        g8.surface.predictor_state.predictor_program,
        g8.semantic_g_packet,
        g8.post_g8_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="post-G8 A requires"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            bare_g_with_post_a,
            causal_surface=g8.causal_surface,
        )


def test_composite_g8_inner_crc_post_a_crc_and_foreign_binding_fail_closed() -> None:
    fixture = _real_g8_fixture_n2()
    corrupted_g = fixture.composite_g_packet[:-1] + bytes([fixture.composite_g_packet[-1] ^ 1])
    bad_g_archive = pga.build_taskspace_monolithic_pga_archive(
        fixture.surface.predictor_state.predictor_program,
        corrupted_g,
        fixture.post_g8_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="composite G failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            bad_g_archive,
            causal_surface=fixture.causal_surface,
        )

    corrupted_a = fixture.post_g8_a_packet[:-1] + bytes([fixture.post_g8_a_packet[-1] ^ 1])
    bad_a_archive = pga.build_taskspace_monolithic_pga_archive(
        fixture.surface.predictor_state.predictor_program,
        fixture.composite_g_packet,
        corrupted_a,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="post-G8 A failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            bad_a_archive,
            causal_surface=fixture.causal_surface,
        )

    foreign = _real_g8_fixture_n2(126)
    foreign_a_archive = pga.build_taskspace_monolithic_pga_archive(
        fixture.surface.predictor_state.predictor_program,
        fixture.composite_g_packet,
        foreign.post_g8_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="post-G8 A failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            foreign_a_archive,
            causal_surface=fixture.causal_surface,
        )


def test_g8_receiver_receipt_rejects_truth_numeric_and_nested_receipt_smuggling() -> None:
    fixture = _real_g8_fixture_n2()
    receipt = pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        fixture.archive,
        causal_surface=fixture.causal_surface,
    ).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["post_g8_y1_unchanged_by_a"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="truth labels"):
        pga.parse_taskspace_monolithic_pga_g8_receiver_receipt(forged_truth)

    body = json.loads(receipt.to_receipt_bytes())
    body["archive_bytes"] = float(body["archive_bytes"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact integers"):
        pga.parse_taskspace_monolithic_pga_g8_receiver_receipt(forged_number)

    body = json.loads(receipt.to_receipt_bytes())
    body["g8_receipt"]["output_camera_y1_sha256"] = "0" * 64
    forged_nested = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match=r"nested g8_receipt|repair custody"):
        pga.parse_taskspace_monolithic_pga_g8_receiver_receipt(forged_nested)


@pytest.mark.parametrize("with_g8", (False, True), ids=("pass-no-g8", "pass-then-g8"))
def test_production_pass_receiver_closes_optional_g8_and_cached_p(
    monkeypatch: pytest.MonkeyPatch,
    with_g8: bool,
) -> None:
    fixture = _real_pass_fixture_n2(with_g8)
    cached = pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        fixture.archive,
        causal_surface=fixture.causal_surface,
    )
    assert type(cached) is pga.DecodedTaskspaceMonolithicPGAPassV1
    _install_cached_causal_decode(monkeypatch, fixture.causal_surface, fixture.runtime)
    default = pga.receive_ep725_taskspace_monolithic_pga_archive(
        fixture.archive,
        predictor_runtime=fixture.runtime,
        pair_count=2,
    )
    assert type(default) is pga.DecodedTaskspaceMonolithicPGAPassV1
    assert cached.receipt.to_receipt_bytes() == default.receipt.to_receipt_bytes()
    assert np.array_equal(cached.chronological_camera_frames, default.chronological_camera_frames)

    receipt = cached.receipt
    assert receipt.sections[0].payload_sha256 == fixture.surface.predictor_state.predictor_program_sha256
    assert receipt.sections[1].payload_sha256 == _sha256(fixture.pass_g_packet)
    assert receipt.sections[2].payload_sha256 == _sha256(fixture.conditional_a_packet)
    assert receipt.pass_g_receipt == fixture.decoded_pass_g.receipt
    assert receipt.conditional_a_receipt == fixture.decoded_conditional_a.receipt
    assert receipt.optional_g8_repair_present is with_g8
    assert receipt.pass_g_receipt.mode == ("PASS_THEN_G8_V1" if with_g8 else "PASS_NO_G8_V1")
    assert (receipt.pass_g_receipt.repair_packet_sha256 is not None) is with_g8
    assert receipt.conditional_a_receipt.source.pass_g_mode == receipt.pass_g_receipt.mode
    assert receipt.conditional_a_receipt.source.pass_g_envelope_sha256 == receipt.sections[1].payload_sha256
    assert np.array_equal(cached.camera_y1, fixture.decoded_pass_g.conditional_camera_y1)
    assert receipt.exact_nonempty_pass_semantic_g_strictly_parsed is True
    assert receipt.exact_conditional_a_strictly_parsed_and_source_bound is True
    assert receipt.conditional_y1_unchanged_by_a is True
    assert receipt.all_counted_p_g_a_semantically_consumed is True
    assert receipt.through_r_target_realization_debt_closed is False
    assert receipt.scorer_invoked is False
    assert receipt.exact_score_claim is False
    assert receipt.candidate_archive_eligible is False
    assert pga.parse_taskspace_monolithic_pga_pass_receiver_receipt(receipt.to_receipt_bytes()) == receipt


def test_production_pass_receiver_rejects_crossed_domains_corruption_and_substituted_p() -> None:
    fixture = _real_pass_fixture_n2(False)
    legacy = _real_a3_fixture_n2()
    exact_g8 = _real_g8_fixture_n2()
    state = fixture.surface.predictor_state

    pass_g_with_legacy_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        fixture.pass_g_packet,
        legacy.pass_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="requires production conditional A"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            pass_g_with_legacy_a,
            causal_surface=fixture.causal_surface,
        )

    exact_g_with_pass_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        exact_g8.semantic_g_packet,
        fixture.conditional_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="requires counted PASS"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            exact_g_with_pass_a,
            causal_surface=fixture.causal_surface,
        )

    corrupted_g = fixture.pass_g_packet[:-1] + bytes([fixture.pass_g_packet[-1] ^ 1])
    bad_g = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        corrupted_g,
        fixture.conditional_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="PASS-G failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            bad_g,
            causal_surface=fixture.causal_surface,
        )

    corrupted_a = fixture.conditional_a_packet[:-1] + bytes([fixture.conditional_a_packet[-1] ^ 1])
    bad_a = pga.build_taskspace_monolithic_pga_archive(
        state.predictor_program,
        fixture.pass_g_packet,
        corrupted_a,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="conditional A failed"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            bad_a,
            causal_surface=fixture.causal_surface,
        )

    changed_p = bytearray(state.predictor_program)
    changed_p[-1] ^= 1
    substituted_p = pga.build_taskspace_monolithic_pga_archive(
        bytes(changed_p),
        fixture.pass_g_packet,
        fixture.conditional_a_packet,
    ).selected.archive_bytes
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="directory-owned P differs"):
        pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
            substituted_p,
            causal_surface=fixture.causal_surface,
        )


def test_pass_receiver_receipt_rejects_truth_numeric_and_nested_smuggling() -> None:
    fixture = _real_pass_fixture_n2(False)
    receipt = pga.receive_ep725_taskspace_monolithic_pga_archive_from_causal_surface(
        fixture.archive,
        causal_surface=fixture.causal_surface,
    ).receipt
    body = json.loads(receipt.to_receipt_bytes())
    body["conditional_y1_unchanged_by_a"] = False
    forged_truth = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="truth labels"):
        pga.parse_taskspace_monolithic_pga_pass_receiver_receipt(forged_truth)

    body = json.loads(receipt.to_receipt_bytes())
    body["archive_bytes"] = float(body["archive_bytes"])
    forged_number = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="exact integers"):
        pga.parse_taskspace_monolithic_pga_pass_receiver_receipt(forged_number)

    body = json.loads(receipt.to_receipt_bytes())
    body["conditional_a_receipt"]["source"]["pass_g_envelope_sha256"] = "0" * 64
    forged_nested = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(pga.TaskspaceMonolithicPGAError, match="nested conditional_a_receipt"):
        pga.parse_taskspace_monolithic_pga_pass_receiver_receipt(forged_nested)
