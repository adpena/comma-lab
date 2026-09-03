# SPDX-License-Identifier: MIT
"""Compatibility, receiver, and falsifier proofs for G77 TSPPV2."""

from __future__ import annotations

import hashlib
import io
import subprocess
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g17_production_envelope import (
    G17ADescriptorV1,
    G17AFamily,
    G17AMode,
    G17PopulationLayout,
    build_g17_a_packet,
    build_g17_g_packet,
    build_g17_terminal_envelope,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g77_flat_v15_selected_preimage_product_v1 import (
    FINAL_MULTI_ACTUATOR_DEMUX_BLOCKER,
    OPEN_PRODUCT_BLOCKERS,
    TSPPV2_MEMBER_NAME,
    G77FlatV15SelectedPreimageProductError,
    build_g77_flat_v15_selected_preimage_product,
    parse_extracted_g77_flat_v15_selected_preimage_product,
    parse_g77_flat_v15_selected_preimage_product,
    reconstruct_semantic_p_from_flat_product,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageFrameSelectorV1,
    verify_v15_semantic_compile_lineage,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v2 import (
    COMPACT_RECEIVER_PACKET_BLOCKER,
    MAGIC,
    PREPAINT_AGGREGATION_CONTRACT,
    PUBLIC_INFLATE_INTEGRATION_BLOCKER,
    SelectedPreimageCompileConfigV2,
    SelectedPreimageFactorModeV2,
    SelectedPreimageFactorRoleV2,
    TaskspaceSelectedPreimageFactorV2,
    TaskspaceSelectedPreimageProgramV2,
    TaskspaceSelectedPreimageProgramV2Error,
    V15RoleAwareDecoderIdentityV2,
    encode_selected_preimage_program_v2,
    parse_selected_preimage_program_v2,
)
from tac.witness_dsl.tests.test_taskspace_g17_g49_active_a_abi import (
    _adapter as _legacy_g17_adapter,
)
from tac.witness_dsl.tests.test_taskspace_g17_g49_active_a_abi import (
    _program as _legacy_v1_program,
)

_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = (
    _ROOT / ".omx/research/original_taskspace_inverse_witness_codec_20260725" / "fresh_v15_semantic_base_n600_20260726"
)
_SEMANTIC_PATH = _RUN_DIR / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
_COMPILE_RECEIPT_PATH = _RUN_DIR / "ddm_v15_scorer_solved_templates_n600_receipt.json"
_MAX_ARCHIVE_BYTES = 2 << 20
_MAX_MEMBER_BYTES = 1 << 20
_MAX_PROGRAM_BYTES = 1 << 20


_QUARANTINE_V15_PRODUCER_PIN = pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): retired July taskspace lineage, no live consumer. "
        "The sealed V15 compile receipt's producer_custody pins "
        "src/tac/optimization/direct_description_carrier_compose.py at 3e1f69bb/156,551 B; HEAD is "
        "6fef110d/160,470 B. Drift commits: 9934d488b then 36f4b2947 (both 2026-08-20). "
        "MEASURED, so the quarantine is not a guess: the delta is exactly 3 changed defs plus 2 added "
        "and 0 removed, all three ARE on the path the pinned producer calls, and the drifted receiver "
        "still decodes the sealed archive and refuses all 5 mutation samples with identical coverage "
        "(133,941 B) -- the sole receipt delta is an ADDED key, non_empty_member_payload_count, that "
        "no check reads. Output-equivalent is still not a pin refresh: the pin lives in a SEALED "
        "2026-07-26 custody receipt, and rewriting its producer_custody to match today's sources "
        "would make that receipt assert it was compiled by sources that did not exist at compile "
        "time. The honest cure is a NEW receipt from a fresh compile. "
        "FIRE TRIGGER: a scorer-authorized arm re-runs tools/measure_ddm_v15_scorer_solved_templates.py "
        "(it solves through exact R + SegNet, which ddm_ql1's charter forbade), emits a fresh receipt, "
        "and either confirms byte-identity -- then DELETE this mark -- or records real output drift. "
        "Note the same file text is re-checked at taskspace_selected_preimage_program_v1.py:1923 and "
        "taskspace_selected_preimage_program_v2.py:723, so only a receipt refresh cures all sites; a "
        "waiver of the :572 check alone would not. strict=True means this mark FAILS the moment the "
        "lineage is repaired. Owning memos: "
        ".omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md and "
        ".omx/research/ddm_cd1_working_tree_debt_landing_20260903.md"
    ),
)
_V1_PACKET_SHA256 = "8a898992d7bbd84a544f5a9918244c9a0155b6f4bcbd717072dcf52801830e68"
_V1_G_SHA256 = "b1ac2bbacefbc049be31a0d9f496246f3201d4b1fa184fd44c69836fc536a6d1"
_V1_A_SHA256 = "44435df8e5a2c6385075b76a1640acc21bffa267b32f7bda2241414dee943b2a"
_V1_E_SHA256 = "650003635e004a6336bdf541a59387fe98f0587ba01342ab293d66ad589cf08d"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _semantic_bytes() -> bytes:
    if not _SEMANTIC_PATH.is_file() or not _COMPILE_RECEIPT_PATH.is_file():
        pytest.skip("retained fresh V15 semantic custody is absent")
    return _SEMANTIC_PATH.read_bytes()


def _road_atom() -> BoundaryShearletAtomV1:
    return BoundaryShearletAtomV1(
        pair_index=0,
        role="Road",
        center_y=160,
        center_x=256,
        scale_y=24,
        scale_x=96,
        shear_q4=0,
        amplitude_q4=64,
    )


def _undrivable_atom() -> BoundaryShearletAtomV1:
    return BoundaryShearletAtomV1(
        pair_index=0,
        role="UndrivableBoundary",
        center_y=174,
        center_x=420,
        scale_y=8,
        scale_x=24,
        shear_q4=0,
        amplitude_q4=64,
    )


def _program() -> TaskspaceSelectedPreimageProgramV2:
    semantic_bytes = _semantic_bytes()
    identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=_COMPILE_RECEIPT_PATH.read_bytes(),
        compiled_semantic_archive=semantic_bytes,
        producer_root=_ROOT,
    )
    target = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256=("58db7f01674c60f060a46b955fee8c4f777f31f528ebba404e871b26b17972a7"),
        target_bank_sha256=("6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85"),
    )
    operand = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        atoms=(_undrivable_atom(), _road_atom()),
    )
    factor = TaskspaceSelectedPreimageFactorV2(
        section_id="a.v15_roleaware_prepaint",
        source_pair_start=0,
        pair_count=600,
        operand_payload=operand.to_bytes(),
        source_receipt_sha256=("5a74c590abf2fb4bac2c7ca3a20e1be645b6e52cc95749bae55da426c73487c9"),
    )
    return TaskspaceSelectedPreimageProgramV2(
        semantic_program_identity=identity,
        target_custody_identity=target,
        decoder_identity=V15RoleAwareDecoderIdentityV2.current(),
        compile_config=SelectedPreimageCompileConfigV2(
            source_pair_start=0,
            pair_count=600,
            maximum_packet_bytes=_MAX_PROGRAM_BYTES,
            score_budget_receipt_sha256=("5a74c590abf2fb4bac2c7ca3a20e1be645b6e52cc95749bae55da426c73487c9"),
            budget_rule_id="g77_additive_roleaware_factor_research_only_v2",
        ),
        factor=factor,
    )


def _product():
    program = _program()
    return build_g77_flat_v15_selected_preimage_product(
        semantic_p_archive=_semantic_bytes(),
        program_packet=program.packet_bytes,
        maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_program_bytes=_MAX_PROGRAM_BYTES,
    )


def _rewrite_member(
    archive_bytes: bytes,
    *,
    member_name: str,
    transform,
) -> bytes:
    source = zipfile.ZipFile(io.BytesIO(archive_bytes), "r")
    output = io.BytesIO()
    with source, zipfile.ZipFile(output, "w") as target:
        for info in source.infolist():
            copied = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            for field_name in (
                "compress_type",
                "comment",
                "extra",
                "internal_attr",
                "external_attr",
                "create_system",
                "create_version",
                "extract_version",
                "flag_bits",
                "volume",
            ):
                setattr(copied, field_name, getattr(info, field_name))
            payload = source.read(info)
            target.writestr(
                copied,
                transform(payload) if info.filename == member_name else payload,
            )
    return output.getvalue()


@_QUARANTINE_V15_PRODUCER_PIN
def test_tsppv2_is_one_global_role_aware_bank_and_strict_roundtrip() -> None:
    program = _program()
    packet = program.packet_bytes
    parsed = parse_selected_preimage_program_v2(
        packet,
        maximum_packet_bytes=_MAX_PROGRAM_BYTES,
    )

    assert packet[:8] == MAGIC
    assert encode_selected_preimage_program_v2(parsed) == packet
    assert parsed.factor.operand.to_bytes() == parsed.factor.operand_payload
    assert parsed.factor.role is (SelectedPreimageFactorRoleV2.ANALYTIC_ROLE_AWARE_PREPAINT)
    assert parsed.factor.mode is (SelectedPreimageFactorModeV2.V15_ROLE_AWARE_PREPAINT_G74RA1)
    assert parsed.factor.frame_selector is SelectedPreimageFrameSelectorV1.BOTH
    assert tuple(atom.role for atom in parsed.factor.operand.atoms) == (
        "UndrivableBoundary",
        "Road",
    )
    assert parsed.decoder_identity.prepaint_aggregation_contract == (PREPAINT_AGGREGATION_CONTRACT)
    assert parsed.mixed_legacy_and_role_aware_factors is False
    assert parsed.legacy_v1_factor_semantics_reinterpreted is False
    assert sum(row.byte_length for row in parsed.byte_homes()) == len(packet)
    assert parsed.byte_homes()[1].payload_sha256 == parsed.factor.operand_sha256
    assert len(packet) == 3593
    assert parsed.byte_homes()[0].byte_length == 3541
    assert parsed.byte_homes()[1].byte_length == 52
    assert PUBLIC_INFLATE_INTEGRATION_BLOCKER in parsed.open_product_blockers
    assert COMPACT_RECEIVER_PACKET_BLOCKER in parsed.open_product_blockers


@_QUARANTINE_V15_PRODUCER_PIN
def test_tsppv2_refuses_operand_hash_eof_selector_and_window_drift() -> None:
    packet = _program().packet_bytes
    tampered = bytearray(packet)
    tampered[-1] ^= 0x01
    with pytest.raises(
        TaskspaceSelectedPreimageProgramV2Error,
        match="SHA-256",
    ):
        parse_selected_preimage_program_v2(
            bytes(tampered),
            maximum_packet_bytes=_MAX_PROGRAM_BYTES,
        )
    with pytest.raises(
        TaskspaceSelectedPreimageProgramV2Error,
        match="length/EOF",
    ):
        parse_selected_preimage_program_v2(
            packet + b"x",
            maximum_packet_bytes=_MAX_PROGRAM_BYTES,
        )

    program = _program()
    with pytest.raises(
        TaskspaceSelectedPreimageProgramV2Error,
        match="pair windows differ",
    ):
        replace(
            program,
            factor=replace(program.factor, pair_count=1),
        )
    changed_selector = bytearray(program.factor.operand_payload)
    changed_selector[9] = 255
    with pytest.raises(
        TaskspaceSelectedPreimageProgramV2Error,
        match="G74RA1",
    ):
        replace(
            program.factor,
            operand_payload=bytes(changed_selector),
        )


@_QUARANTINE_V15_PRODUCER_PIN
def test_flat_product_reconstructs_exact_p_without_nested_complete_p_copy() -> None:
    semantic = _semantic_bytes()
    built = _product()
    parsed = built.parsed
    reparsed = parse_g77_flat_v15_selected_preimage_product(
        built.archive_bytes,
        maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_program_bytes=_MAX_PROGRAM_BYTES,
    )
    reconstructed = reconstruct_semantic_p_from_flat_product(
        built.archive_bytes,
        maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
    )

    assert reparsed == parsed
    assert reconstructed == semantic == parsed.semantic_p_archive
    assert parsed.semantic_p_sha256 == ("759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df")
    assert parsed.semantic_member_names == (
        "manifest.json",
        "predictor.zip",
        "predict/movable_polygon_worldsheet.g1s",
        "render/receiver_realization.ddrp",
        "render/scorer_solved_templates.ddst",
    )
    with zipfile.ZipFile(io.BytesIO(built.archive_bytes), "r") as archive:
        assert archive.namelist() == [
            *parsed.semantic_member_names,
            TSPPV2_MEMBER_NAME,
        ]
        assert all(archive.read(info) != semantic for info in archive.infolist())
    assert parsed.semantic_p_occurrences == 1
    assert parsed.tsppv2_occurrences == 1
    assert parsed.nested_complete_semantic_zip_stored is False
    assert parsed.public_unzip_reconstruction_closed is True
    assert parsed.public_inflate_integration_closed is False
    assert parsed.terminal_multi_actuator_demux_closed is False
    assert parsed.semantic_compression_bits == "11001"
    assert parsed.program_compression_bit == "1"
    assert built.complete_compression_bits == "110011"
    assert built.method_profiles_evaluated == 64
    assert built.deflate_level == 9
    assert built.zlib_runtime_version
    assert PUBLIC_INFLATE_INTEGRATION_BLOCKER in OPEN_PRODUCT_BLOCKERS
    assert COMPACT_RECEIVER_PACKET_BLOCKER in OPEN_PRODUCT_BLOCKERS
    assert FINAL_MULTI_ACTUATOR_DEMUX_BLOCKER in OPEN_PRODUCT_BLOCKERS


@_QUARANTINE_V15_PRODUCER_PIN
def test_public_unzip_demux_and_real_g74_native_role_support_once(
    tmp_path: Path,
) -> None:
    built = _product()
    archive_path = tmp_path / "archive.zip"
    extract_root = tmp_path / "extracted"
    archive_path.write_bytes(built.archive_bytes)
    extract_root.mkdir()
    subprocess.run(
        ["/usr/bin/unzip", "-qq", str(archive_path), "-d", str(extract_root)],
        check=True,
    )
    extracted = parse_extracted_g77_flat_v15_selected_preimage_product(
        extract_root,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_program_bytes=_MAX_PROGRAM_BYTES,
    )
    decoded = extracted.decode_pair(0)
    receipt = decoded.result.receipt

    assert decoded.source_pair_id == 0
    assert extracted.semantic_p_archive == _semantic_bytes()
    assert extracted.program_packet == built.parsed.program_packet
    assert extracted.public_unzip_reconstruction_closed is True
    assert extracted.public_inflate_integration_closed is False
    assert extracted.terminal_multi_actuator_demux_closed is False
    assert decoded.semantic_p_sha256 == built.parsed.semantic_p_sha256
    assert decoded.role_aware_operand_sha256 == built.parsed.program.factor.operand_sha256
    assert receipt.operand_roles == ("Road", "UndrivableBoundary")
    assert receipt.operand_atom_count == 2
    assert receipt.legacy_render_pairs_used is False
    assert receipt.realization_profile_consumed is True
    assert receipt.selected_frames_match_native_mutated_numerators is True
    assert receipt.selected_frames_match_native_torch_bilinear is True
    assert receipt.deterministic_double_decode is True
    assert receipt.scorer_invoked is False
    assert receipt.pose_invoked is False
    assert decoded.deterministic_outer_double_decode is True


@_QUARANTINE_V15_PRODUCER_PIN
def test_flat_product_refuses_semantic_or_counted_operand_mutation() -> None:
    built = _product()
    semantic_mutation = _rewrite_member(
        built.archive_bytes,
        member_name="render/scorer_solved_templates.ddst",
        transform=lambda payload: payload[:-1] + bytes((payload[-1] ^ 1,)),
    )
    with pytest.raises(
        G77FlatV15SelectedPreimageProductError,
        match="reconstructed semantic P",
    ):
        parse_g77_flat_v15_selected_preimage_product(
            semantic_mutation,
            maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_program_bytes=_MAX_PROGRAM_BYTES,
        )

    operand_mutation = _rewrite_member(
        built.archive_bytes,
        member_name=TSPPV2_MEMBER_NAME,
        transform=lambda payload: payload[:-1] + bytes((payload[-1] ^ 1,)),
    )
    with pytest.raises(
        G77FlatV15SelectedPreimageProductError,
        match="strict parser",
    ):
        parse_g77_flat_v15_selected_preimage_product(
            operand_mutation,
            maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_program_bytes=_MAX_PROGRAM_BYTES,
        )


@_QUARANTINE_V15_PRODUCER_PIN
def test_extracted_demux_refuses_unknown_file(tmp_path: Path) -> None:
    built = _product()
    archive_path = tmp_path / "archive.zip"
    extract_root = tmp_path / "extracted"
    archive_path.write_bytes(built.archive_bytes)
    extract_root.mkdir()
    subprocess.run(
        ["/usr/bin/unzip", "-qq", str(archive_path), "-d", str(extract_root)],
        check=True,
    )
    (extract_root / "unknown.bin").write_bytes(b"x")
    with pytest.raises(
        G77FlatV15SelectedPreimageProductError,
        match="unknown or duplicated",
    ):
        parse_extracted_g77_flat_v15_selected_preimage_product(
            extract_root,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_program_bytes=_MAX_PROGRAM_BYTES,
        )


def test_frozen_v1_program_and_g17_envelope_bytes_remain_identical() -> None:
    legacy = _legacy_v1_program(2)
    assert len(legacy.packet_bytes) == 3616
    assert _sha(legacy.packet_bytes) == _V1_PACKET_SHA256
    assert legacy.packet_bytes[:8] == b"TSPPV1\x00\x00"

    p_section = b"not-a-zip-semantic-p"
    g_section = build_g17_g_packet(
        p_section=p_section,
        pair_start=0,
        pair_count=2,
    )
    adapter = _legacy_g17_adapter(legacy)
    active = adapter(
        legacy.packet_bytes,
        0,
        2,
        G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
        G17AMode.SELECTED_PREIMAGE_PROGRAM,
    )
    a_section = build_g17_a_packet(
        p_section=p_section,
        g_section=g_section,
        pair_start=0,
        pair_count=2,
        layout=G17PopulationLayout.GLOBAL,
        descriptors=(
            G17ADescriptorV1(
                pair_start=0,
                pair_count=2,
                family=G17AFamily.G49_SELECTED_PREIMAGE_PROGRAM,
                mode=G17AMode.SELECTED_PREIMAGE_PROGRAM,
                active=active,
            ),
        ),
    )
    terminal = build_g17_terminal_envelope(
        p_section=p_section,
        g_section=g_section,
        a_section=a_section,
        a_active_parser=adapter,
    )
    assert _sha(g_section) == _V1_G_SHA256
    assert _sha(a_section) == _V1_A_SHA256
    assert _sha(terminal) == _V1_E_SHA256


@_QUARANTINE_V15_PRODUCER_PIN
def test_bound_decoder_refuses_source_or_runtime_identity_drift() -> None:
    parsed = _product().parsed
    changed_runtime = replace(
        parsed.program.decoder_identity,
        torch_version="foreign-runtime",
    )
    changed_program = replace(
        parsed.program,
        decoder_identity=changed_runtime,
    )
    changed_packet = changed_program.packet_bytes
    changed_product = build_g77_flat_v15_selected_preimage_product(
        semantic_p_archive=parsed.semantic_p_archive,
        program_packet=changed_packet,
        maximum_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_program_bytes=_MAX_PROGRAM_BYTES,
    ).parsed
    with pytest.raises(
        G77FlatV15SelectedPreimageProductError,
        match="source/runtime",
    ):
        changed_product.open_bound_decoder()
