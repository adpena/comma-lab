# SPDX-License-Identifier: MIT
"""Exact wire and real fresh-G82 execution proofs for G88 conditional Y0."""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.xi_pose_coder import serialize_xi_payload
from tac.optimization.direct_description_carrier_compose import (
    REALIZATION_PAINT_ORDER,
    BoundaryShearletAtomV1,
)
from tac.witness_dsl import taskspace_g88_population_conditional_y0_pvsa_v1 as g88
from tac.witness_dsl.taskspace_counted_xip2_chronological_a3 import (
    NUMERIC_REFERENCE_ID,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_outer_archive_codec import OuterArchiveEncoding
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    build_compact_pvsa_archive,
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
_BASE_MEMBER_SHA256 = "d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31"
_G74_OPERAND_SHA256 = "5616799adc0d2ab942f37a20b070f7a0fa48119771e8f1b56c1f45e2605306ca"
_G85_ROOT = Path("/Volumes/VertigoDataTier/pact/g85_pvsa_public_receiver_20260727_r1")
_MAX_MEMBER_BYTES = 2 << 20
_MAX_SECTION_BYTES = 1 << 20


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _semantic() -> bytes:
    if not _SEMANTIC_PATH.is_file():
        pytest.skip("retained fresh V15 semantic P is absent")
    payload = _SEMANTIC_PATH.read_bytes()
    assert len(payload) == 133_941
    assert _sha(payload) == _SEMANTIC_SHA256
    return payload


def _g74_operand() -> bytes:
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
    payload = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        atoms=atoms,
    ).to_bytes()
    assert len(payload) == 52
    assert _sha(payload) == _G74_OPERAND_SHA256
    return payload


@pytest.fixture(scope="module")
def base_member() -> bytes:
    retained = _G85_ROOT / "archive/0.bin"
    if retained.is_file():
        payload = retained.read_bytes()
        if len(payload) == 133_363 and _sha(payload) == _BASE_MEMBER_SHA256:
            return payload
    built = build_compact_pvsa_archive(
        semantic_p_archive=_semantic(),
        actuator_payloads=(_g74_operand(),),
        maximum_semantic_archive_bytes=_MAX_MEMBER_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )
    assert built.selected.member_sha256 == _BASE_MEMBER_SHA256
    return built.selected.member_bytes


def _copy_operand(base_member: bytes, *pair_ids: int) -> g88.PopulationConditionalOperandV1:
    return g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha(base_member),
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=tuple(g88.ConditionalY0ControlV1.copy_conditional_y1(pair_id) for pair_id in pair_ids),
    )


def _reseal_operand_with_mode(payload: bytes, mode_wire: int) -> bytes:
    header = list(g88._OPERAND_HEADER.unpack_from(payload))
    body_start = g88._OPERAND_HEADER.size
    body_stop = body_start + header[9]
    body = bytearray(payload[body_start:body_stop])
    xip2_payload = payload[body_stop : -g88._CRC32.size]
    body[2] = mode_wire
    header[10] = bytes.fromhex(_sha(bytes(body)))
    prefix = g88._OPERAND_HEADER.pack(*header) + bytes(body) + xip2_payload
    return prefix + struct.pack(">I", zlib.crc32(prefix) & 0xFFFFFFFF)


def _synthetic_mechanism_xip2() -> bytes:
    """One nonzero row; test-only and never a candidate/custodied pose claim."""

    q_codes = np.zeros((600, 6), dtype=np.int16)
    q_codes[0, 0] = 32
    scales = np.full((6,), np.float32(1e-3), dtype=np.float32)
    return serialize_xi_payload(q_codes, scales, coder="delta_ar")


def test_n600_sparse_operand_is_canonical_and_copy_has_exact_byte_price(
    base_member: bytes,
) -> None:
    operand = _copy_operand(base_member, 0)
    payload = operand.to_bytes()
    parsed = g88.parse_population_conditional_operand(
        payload,
        expected_sha256=_sha(payload),
    )

    assert parsed == operand
    assert parsed.to_bytes() == payload
    assert parsed.source_pair_start == 0
    assert parsed.pair_count == 600
    assert parsed.active_pair_count == 1
    assert parsed.pass_pair_count == 599
    assert parsed.control_for_pair(0).mode is (g88.ConditionalY0ModeV1.COPY_CONDITIONAL_Y1)
    assert parsed.control_for_pair(599) is None
    assert len(payload) == 198
    assert g88.PASS_POLICY_ID == "DEFAULT_PASS_OR_XIP2_WITH_SPARSE_TYPED_OVERRIDES_V1"


def test_successor_member_and_both_outer_encodings_parse_back_exactly(
    base_member: bytes,
) -> None:
    operand = _copy_operand(base_member, 0)
    built = g88.build_population_conditional_pvsa_archive(
        base_pvsa_member_bytes=base_member,
        conditional_operand_bytes=operand.to_bytes(),
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )

    assert built.stored == built.deflated == built.selected
    assert built.selected.base_pvsa_member_bytes == base_member
    assert built.selected.conditional_operand == operand
    assert built.selected.member_bytes.startswith(g88.SUCCESSOR_MAGIC)
    assert len(built.selected.member_bytes) == 133_646
    assert len(built.selected.member_bytes) - len(base_member) == 283
    assert built.outer_build.selected.encoding is OuterArchiveEncoding.DEFLATED
    assert built.outer_build.selected.archive_nbytes < len(built.selected.member_bytes)
    assert g88.PUBLIC_RUNTIME_BLOCKER in built.selected.open_blockers
    assert g88.POSE_AUTHORITY_BLOCKER in built.selected.open_blockers
    assert g88.FRESH_XIP2_CUSTODY_BLOCKER in built.selected.open_blockers
    assert (
        g88.encode_population_conditional_pvsa_member(
            base_pvsa_member_bytes=base_member,
            conditional_operand_bytes=operand.to_bytes(),
        )
        == built.selected.member_bytes
    )


def test_operand_and_successor_refuse_aliases_unknown_modes_and_mutation(
    base_member: bytes,
) -> None:
    operand = _copy_operand(base_member, 0)
    payload = operand.to_bytes()

    with pytest.raises(g88.PopulationConditionalPVSAError, match="alias"):
        g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha(base_member),
            semantic_p_sha256=_SEMANTIC_SHA256,
            controls=(
                g88.ConditionalY0ControlV1(
                    source_pair_id=0,
                    mode=g88.ConditionalY0ModeV1.PASS_P0,
                ),
            ),
        )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="unique ascending"):
        g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha(base_member),
            semantic_p_sha256=_SEMANTIC_SHA256,
            controls=(
                g88.ConditionalY0ControlV1.copy_conditional_y1(1),
                g88.ConditionalY0ControlV1.copy_conditional_y1(0),
            ),
        )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="unknown"):
        g88.parse_population_conditional_operand(_reseal_operand_with_mode(payload, 255))
    xip2_control = g88.ConditionalY0ControlV1(
        source_pair_id=0,
        mode=g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
    )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="requires one exact"):
        g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha(base_member),
            semantic_p_sha256=_SEMANTIC_SHA256,
            controls=(xip2_control,),
        )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="non-XIP2"):
        g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha(base_member),
            semantic_p_sha256=_SEMANTIC_SHA256,
            controls=(g88.ConditionalY0ControlV1.copy_conditional_y1(0),),
            xip2_payload=_synthetic_mechanism_xip2(),
        )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="requires one exact"):
        g88.PopulationConditionalOperandV1(
            base_pvsa_member_sha256=_sha(base_member),
            semantic_p_sha256=_SEMANTIC_SHA256,
            controls=(xip2_control,),
            xip2_payload=np.zeros((600, 6), dtype=">i2").tobytes(),
        )
    global_xip2 = g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha(base_member),
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=(),
        default_mode=g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
        xip2_payload=_synthetic_mechanism_xip2(),
        pitch=float(np.float32(-0.03)),
    ).to_bytes()
    xip2_header = g88._OPERAND_HEADER.unpack_from(global_xip2)
    xip2_start = g88._OPERAND_HEADER.size + xip2_header[9]
    digest_corrupt = bytearray(global_xip2)
    digest_corrupt[xip2_start] ^= 1
    prefix = bytes(digest_corrupt[: -g88._CRC32.size])
    digest_corrupt[-g88._CRC32.size :] = struct.pack(
        ">I",
        zlib.crc32(prefix) & 0xFFFFFFFF,
    )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="XIP2 SHA-256"):
        g88.parse_population_conditional_operand(bytes(digest_corrupt))

    corrupt = bytearray(payload)
    corrupt[-1] ^= 1
    with pytest.raises(g88.PopulationConditionalPVSAError, match="CRC32"):
        g88.parse_population_conditional_operand(bytes(corrupt))
    with pytest.raises(g88.PopulationConditionalPVSAError, match=r"EOF|truncated"):
        g88.parse_population_conditional_operand(payload[:-1])

    wrong = g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256="1" * 64,
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=(g88.ConditionalY0ControlV1.copy_conditional_y1(0),),
    )
    member = g88.encode_population_conditional_pvsa_member(
        base_pvsa_member_bytes=base_member,
        conditional_operand_bytes=wrong.to_bytes(),
    )
    with pytest.raises(g88.PopulationConditionalPVSAError, match="different base"):
        g88.parse_population_conditional_pvsa_member(
            member,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )
    mutated_member = bytearray(
        g88.encode_population_conditional_pvsa_member(
            base_pvsa_member_bytes=base_member,
            conditional_operand_bytes=payload,
        )
    )
    mutated_member[-1] ^= 1
    with pytest.raises(g88.PopulationConditionalPVSAError, match="CRC32"):
        g88.parse_population_conditional_pvsa_member(
            bytes(mutated_member),
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )


def _g85_double_decode_pairs(pair_count: int) -> tuple[np.ndarray, np.ndarray]:
    input_path = _G85_ROOT / "receipts/input.json"
    first_path = _G85_ROOT / "decode_a/0.raw"
    second_path = _G85_ROOT / "decode_b/0.raw"
    if not input_path.is_file() or not first_path.is_file() or not second_path.is_file():
        pytest.skip("fresh G85 full-n600 public-receiver double decode is absent")
    custody = json.loads(input_path.read_text())
    assert custody == {
        "archive_bytes": 129_392,
        "archive_sha256": ("b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd"),
        "member_bytes": 133_363,
        "member_sha256": _BASE_MEMBER_SHA256,
        "research_only": True,
        "schema": "tac.g85_exact_g82_input_materialization.v1",
        "score_claim": False,
        "source": "fresh committed G82 lowering; no historical payload reuse",
    }
    exact_bytes = 600 * 2 * 874 * 1164 * 3
    assert first_path.stat().st_size == exact_bytes
    assert second_path.stat().st_size == exact_bytes
    shape = (600, 2, 874, 1164, 3)
    first = np.asarray(np.memmap(first_path, dtype=np.uint8, mode="r", shape=shape)[:pair_count]).copy()
    second = np.asarray(np.memmap(second_path, dtype=np.uint8, mode="r", shape=shape)[:pair_count]).copy()
    return first, second


def test_real_fresh_g82_pair_zero_copy_changes_y0_and_preserves_exact_y1(
    base_member: bytes,
) -> None:
    first, second = _g85_double_decode_pairs(1)
    operand = _copy_operand(base_member, 0)
    result = g88.apply_population_conditional_to_decoded_batch(
        operand=operand,
        first_base_camera_pairs=first,
        second_base_camera_pairs=second,
        local_pair_ids=(0,),
    )

    assert _sha(first[0].tobytes()) == ("caf69dade383564ef8123149d193052b8b5b641711fed232b25dd67b29af25db")
    assert _sha(first[0, 0].tobytes()) == ("754ce88b494bfbc3bd560b23ee26cdbccffb10d5829fa09d882e04f918aa9126")
    assert _sha(first[0, 1].tobytes()) == ("65ca46b182ef52d4cedffb56ec48576bd19610802b60f88027a9e7e46158a037")
    assert result.active_pair_ids == (0,)
    assert result.changed_y0_values == 828_605
    assert result.changed_y0_pixels == 429_630
    assert result.deterministic_double_decode is True
    assert np.all(result.owned_y0_values)
    assert result.preserved_unowned_y0_values == 0
    assert np.array_equal(result.camera_pairs[0, 0], first[0, 1])
    assert np.array_equal(result.camera_pairs[0, 1], first[0, 1])


def test_sparse_default_pass_is_real_across_one_bounded_batch(
    base_member: bytes,
) -> None:
    first, second = _g85_double_decode_pairs(2)
    operand = _copy_operand(base_member, 0)
    result = g88.apply_population_conditional_to_decoded_batch(
        operand=operand,
        first_base_camera_pairs=first,
        second_base_camera_pairs=second,
        local_pair_ids=(0, 1),
    )

    assert result.active_pair_ids == (0,)
    assert np.array_equal(result.camera_pairs[1], first[1])
    assert not np.any(result.owned_y0_values[1])
    assert np.array_equal(result.camera_pairs[:, 1], first[:, 1])
    with pytest.raises(g88.PopulationConditionalPVSAError, match="contiguous"):
        g88.apply_population_conditional_to_decoded_batch(
            operand=operand,
            first_base_camera_pairs=first,
            second_base_camera_pairs=second,
            local_pair_ids=(0, 2),
        )


def test_xip2_mode_roundtrips_and_warps_exact_y1_deterministically(
    base_member: bytes,
) -> None:
    first, second = _g85_double_decode_pairs(1)
    pitch = float(np.float32(-0.03))
    operand = g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha(base_member),
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=(),
        default_mode=g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
        xip2_payload=_synthetic_mechanism_xip2(),
        pitch=pitch,
    )
    payload = operand.to_bytes()
    parsed = g88.parse_population_conditional_operand(
        payload,
        expected_sha256=_sha(payload),
    )
    assert parsed.to_bytes() == payload
    assert parsed.pitch == pitch
    assert parsed.transport is not None
    assert parsed.transport.q_codes.shape == (600, 6)
    assert parsed.transport.q_codes.dtype == np.int16
    assert parsed.transport.counted_payload == parsed.xip2_payload
    assert parsed.transport.predictor_program_sha256 == _sha(base_member)
    assert g88.XIP2_NUMERIC_REFERENCE_ID == NUMERIC_REFERENCE_ID
    assert parsed.active_pair_count == 600
    assert parsed.pass_pair_count == 0
    assert parsed.mode_for_pair(599) is g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP

    pass_override = g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha(base_member),
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=(
            g88.ConditionalY0ControlV1(
                source_pair_id=599,
                mode=g88.ConditionalY0ModeV1.PASS_P0,
            ),
        ),
        default_mode=g88.ConditionalY0ModeV1.XIP2_SE3_FRAME0_WARP,
        xip2_payload=_synthetic_mechanism_xip2(),
        pitch=pitch,
    )
    assert pass_override.active_pair_count == 599
    assert pass_override.pass_pair_count == 1
    assert pass_override.mode_for_pair(599) is g88.ConditionalY0ModeV1.PASS_P0

    result = g88.apply_population_conditional_to_decoded_batch(
        operand=parsed,
        first_base_camera_pairs=first,
        second_base_camera_pairs=second,
        local_pair_ids=(0,),
    )
    assert result.active_pair_ids == (0,)
    assert result.deterministic_double_decode is True
    assert result.changed_y0_values > 0
    assert result.changed_y0_pixels > 0
    assert np.all(result.owned_y0_values)
    assert result.preserved_unowned_y0_values == 0
    assert np.array_equal(result.camera_pairs[:, 1], first[:, 1])


def test_real_role_bounded_translation_rgb_preserves_unowned_p0(
    base_member: bytes,
) -> None:
    road_index = REALIZATION_PAINT_ORDER.index("Road")
    deltas = [(0, 0, 0) for _ in REALIZATION_PAINT_ORDER]
    # Deterministically derived from the nonzero channel-wise medians of
    # fresh-G82 pair-0 (P0 - exact corrected Y1): (+1, -1, +1).
    deltas[road_index] = (1, -1, 1)
    operand = g88.PopulationConditionalOperandV1(
        base_pvsa_member_sha256=_sha(base_member),
        semantic_p_sha256=_SEMANTIC_SHA256,
        controls=(
            g88.ConditionalY0ControlV1(
                source_pair_id=0,
                mode=g88.ConditionalY0ModeV1.ROLE_TRANSLATE_RGB,
                role_bits=1 << road_index,
                role_rgb_deltas=tuple(deltas),
            ),
        ),
    )
    successor = g88.parse_population_conditional_pvsa_member(
        g88.encode_population_conditional_pvsa_member(
            base_pvsa_member_bytes=base_member,
            conditional_operand_bytes=operand.to_bytes(),
        ),
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )
    receiver = successor.open_receiver(verify_member_effects=False)
    result = receiver.decode_pair(0)

    assert result.active_pair_ids == (0,)
    assert result.deterministic_double_decode is True
    assert result.changed_y0_values > 0
    assert result.changed_y0_pixels > 0
    assert 0 < np.count_nonzero(result.owned_y0_values) < result.owned_y0_values.size
    assert result.preserved_unowned_y0_values > 0
    assert np.array_equal(
        result.camera_pairs[:, 0][~result.owned_y0_values],
        result.base_camera_pairs[:, 0][~result.owned_y0_values],
    )
    assert np.array_equal(result.camera_pairs[:, 1], result.base_camera_pairs[:, 1])
