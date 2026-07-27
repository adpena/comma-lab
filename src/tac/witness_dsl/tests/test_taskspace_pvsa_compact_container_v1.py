# SPDX-License-Identifier: MIT
"""Exact wire, reconstruction, rate, and decode proofs for compact PVSA V1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_outer_archive_codec import OuterArchiveEncoding
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    ACTUATOR_TRANSITION_DAG_ID,
    CONDITIONAL_Y0_ACTUATOR_BLOCKER,
    MAGIC,
    V1_ACTUATOR_TRANSITION_PREFIX,
    CompactActuatorTypeV1,
    CompactActuatorV1,
    CompactPVSAError,
    build_compact_pvsa_archive,
    encode_compact_pvsa_member,
    parse_compact_pvsa_member,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    SelectedPreimageFrameSelectorV1,
)

_ROOT = Path(__file__).resolve().parents[4]
_SEMANTIC_PATH = (
    _ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "fresh_v15_semantic_base_n600_20260726"
    / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
)
_SEMANTIC_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
_MAX_ARCHIVE_BYTES = 2 << 20
_MAX_MEMBER_BYTES = 2 << 20
_MAX_SECTION_BYTES = 1 << 20


def _semantic() -> bytes:
    if not _SEMANTIC_PATH.is_file():
        pytest.skip("retained fresh V15 semantic P is absent")
    payload = _SEMANTIC_PATH.read_bytes()
    assert len(payload) == 133_941
    assert hashlib.sha256(payload).hexdigest() == _SEMANTIC_SHA256
    return payload


def _operand() -> bytes:
    atoms = (
        BoundaryShearletAtomV1(
            pair_index=0,
            role="UndrivableBoundary",
            center_y=174,
            center_x=420,
            scale_y=8,
            scale_x=24,
            shear_q4=0,
            amplitude_q4=64,
        ),
        BoundaryShearletAtomV1(
            pair_index=0,
            role="Road",
            center_y=160,
            center_x=256,
            scale_y=24,
            scale_x=96,
            shear_q4=0,
            amplitude_q4=64,
        ),
    )
    return RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.Y1,
        atoms=atoms,
    ).to_bytes()


def _build(
    *,
    with_g74: bool = True,
    rich_ir_bytes: int | None = 3_593,
):
    return build_compact_pvsa_archive(
        semantic_p_archive=_semantic(),
        actuator_payloads=((_operand(),) if with_g74 else ()),
        maximum_semantic_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
        rich_compiler_ir_bytes=rich_ir_bytes,
    )


def test_real_p_compacts_to_one_member_and_reconstructs_exact_semantic_zip() -> None:
    built = _build()
    selected = built.outer_build.selected

    assert built.selected.semantic_p_archive == _semantic()
    assert built.selected.semantic_p_sha256 == _SEMANTIC_SHA256
    assert built.selected.member_bytes.startswith(MAGIC)
    assert _semantic() not in built.selected.member_bytes
    assert selected.member_name == "0.bin"
    assert selected.encoding is OuterArchiveEncoding.DEFLATED
    assert selected.archive_nbytes < 130_000
    assert selected.archive_nbytes < len(_semantic()) - 4_500
    assert built.compact_member_bytes == 133_363
    assert built.rich_ir_bytes_avoided == 3_541
    assert built.selected.actuators[0].actuator_type is (CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT)
    assert built.selected.actuators[0].payload == _operand()
    assert CONDITIONAL_Y0_ACTUATOR_BLOCKER in built.selected.open_product_blockers
    assert built.stored == built.deflated == built.selected


def test_semantic_only_baseline_and_g74_have_exact_same_container_marginal() -> None:
    baseline = _build(with_g74=False, rich_ir_bytes=None)
    with_g74 = _build()

    assert baseline.selected.actuators == ()
    assert baseline.selected.semantic_p_archive == with_g74.selected.semantic_p_archive
    assert baseline.compact_member_bytes == 133_306
    assert baseline.outer_build.selected.encoding is OuterArchiveEncoding.DEFLATED
    assert baseline.outer_build.selected.archive_nbytes == 129_335
    assert with_g74.outer_build.selected.archive_nbytes - baseline.outer_build.selected.archive_nbytes == 57

    receiver = baseline.selected.open_receiver(verify_member_effects=True)
    decoder_identity = id(receiver.overlay_decoder)
    base_pair = receiver.decode_pair(0)
    repeat = receiver.decode_pair(0)
    assert base_pair.shape == (2, 874, 1164, 3)
    assert base_pair.flags.writeable is False
    assert base_pair.tobytes() == repeat.tobytes()
    assert id(receiver.overlay_decoder) == decoder_identity
    assert next(receiver.iter_camera_pair_batches(batch_pairs=16)).shape[0] == 16
    with pytest.raises(CompactPVSAError, match="exactly one G74"):
        baseline.selected.decode_g74_pair(0)
    with pytest.raises(CompactPVSAError, match="zero actuators"):
        with_g74.selected.decode_base_pair(0)
    with pytest.raises(CompactPVSAError, match="contiguous"):
        receiver.render_camera_pair_batch((0, 2))
    assert V1_ACTUATOR_TRANSITION_PREFIX == (CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT,)
    assert "FUTURE_CONDITIONAL_Y0_GIVEN_Y1" in ACTUATOR_TRANSITION_DAG_ID


def test_zero_add_remove_actuator_roundtrip_is_exact() -> None:
    with_g74 = _build().selected
    zero_member = encode_compact_pvsa_member(
        semantic_payloads=with_g74.semantic_payloads,
        actuators=(),
    )
    zero = parse_compact_pvsa_member(
        zero_member,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )
    assert zero.actuators == ()
    assert zero.semantic_p_archive == _semantic()
    assert (
        encode_compact_pvsa_member(
            semantic_payloads=zero.semantic_payloads,
            actuators=with_g74.actuators,
        )
        == with_g74.member_bytes
    )


def test_compact_member_parse_reencode_and_exact_byte_homes() -> None:
    built = _build()
    parsed = parse_compact_pvsa_member(
        built.selected.member_bytes,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )
    assert (
        encode_compact_pvsa_member(
            semantic_payloads=parsed.semantic_payloads,
            actuators=parsed.actuators,
        )
        == parsed.member_bytes
    )
    assert sum(len(payload) for payload in parsed.semantic_payloads) == 133_277
    assert len(parsed.actuators[0].payload) == 52
    assert len(parsed.member_bytes) == 133_363


def test_compact_member_refuses_unknown_type_length_eof_and_trailing_bytes() -> None:
    member = bytearray(_build().selected.member_bytes)
    descriptor_offset = 8 + 5 * 4 + 1

    unknown = bytearray(member)
    unknown[descriptor_offset] = 255
    with pytest.raises(CompactPVSAError, match="unknown"):
        parse_compact_pvsa_member(
            bytes(unknown),
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )

    invalid_length = bytearray(member)
    invalid_length[descriptor_offset + 1 : descriptor_offset + 5] = (0).to_bytes(
        4,
        "little",
    )
    with pytest.raises(CompactPVSAError, match="length"):
        parse_compact_pvsa_member(
            bytes(invalid_length),
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )

    with pytest.raises(CompactPVSAError, match=r"escapes|truncated"):
        parse_compact_pvsa_member(
            bytes(member[:-1]),
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )
    with pytest.raises(CompactPVSAError, match="trailing"):
        parse_compact_pvsa_member(
            bytes(member) + b"x",
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )


def test_compact_member_refuses_noncanonical_actuator_sequence() -> None:
    actuator = CompactActuatorV1(
        actuator_type=CompactActuatorTypeV1.G74_ROLE_AWARE_PREPAINT,
        payload=_operand(),
    )
    semantic = _build().selected.semantic_payloads
    with pytest.raises(CompactPVSAError, match=r"duplicated|order"):
        encode_compact_pvsa_member(
            semantic_payloads=semantic,
            actuators=(actuator, actuator),
        )


def test_real_pair_decode_uses_only_compact_g74_operand() -> None:
    parsed = _build().selected
    first = parsed.decode_g74_pair(0, verify_member_effects=True)
    second = parsed.decode_g74_pair(0, verify_member_effects=True)

    assert first.receipt.base_archive_sha256 == _SEMANTIC_SHA256
    assert first.receipt.frame_selector == "Y1"
    assert first.receipt.operand_atom_count == 2
    assert first.receipt.operand_roles == ("Road", "UndrivableBoundary")
    assert first.receipt.selected_frames_match_native_torch_bilinear is True
    assert first.receipt.deterministic_double_decode is True
    assert first.receipt.to_bytes() == second.receipt.to_bytes()
    assert first.camera_pairs.tobytes() == second.camera_pairs.tobytes()
