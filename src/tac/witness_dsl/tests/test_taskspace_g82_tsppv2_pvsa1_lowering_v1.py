# SPDX-License-Identifier: MIT
"""Exact custody, marginal-rate, and decode proofs for G82 lowering."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from tac.optimization.direct_description_carrier_compose import (
    BoundaryShearletAtomV1,
)
from tac.witness_dsl.taskspace_g74_v15_roleaware_overlay_decoder_v1 import (
    RoleAwareBoundaryShearletOperandV1,
)
from tac.witness_dsl.taskspace_g82_tsppv2_pvsa1_lowering_v1 import (
    FULL_N600_BLOCKER,
    OPEN_LOWERING_BLOCKERS,
    TSPPV2PVSA1LoweringError,
    TSPPV2ToPVSA1LoweringV1,
    lower_tsppv2_to_compact_pvsa1,
)
from tac.witness_dsl.taskspace_outer_archive_codec import (
    parse_taskspace_outer_archive,
)
from tac.witness_dsl.taskspace_pvsa_compact_container_v1 import (
    CONDITIONAL_Y0_ACTUATOR_BLOCKER,
    PUBLIC_INFLATE_BLOCKER,
    parse_compact_pvsa_member,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v1 import (
    ScorerTargetCustodyIdentityV1,
    SelectedPreimageFrameSelectorV1,
    verify_v15_semantic_compile_lineage,
)
from tac.witness_dsl.taskspace_selected_preimage_program_v2 import (
    SelectedPreimageCompileConfigV2,
    TaskspaceSelectedPreimageFactorV2,
    TaskspaceSelectedPreimageProgramV2,
    V15RoleAwareDecoderIdentityV2,
    parse_selected_preimage_program_v2,
)

_ROOT = Path(__file__).resolve().parents[4]
_RUN_DIR = (
    _ROOT / ".omx/research/original_taskspace_inverse_witness_codec_20260725" / "fresh_v15_semantic_base_n600_20260726"
)
_SEMANTIC_PATH = _RUN_DIR / "ddm_v15_solved_templates_n600.not_a_candidate.zip.receipt-bytes"
_COMPILE_RECEIPT_PATH = _RUN_DIR / "ddm_v15_scorer_solved_templates_n600_receipt.json"
_MAX_ARCHIVE_BYTES = 2 << 20
_MAX_MEMBER_BYTES = 2 << 20
_MAX_SECTION_BYTES = 1 << 20
_MAX_PROGRAM_BYTES = 1 << 20


_QUARANTINE_V15_PRODUCER_PIN = pytest.mark.xfail(
    strict=True,
    reason=(
        "QUARANTINED 2026-09-03 (ddm_ql1): retired July taskspace lineage, no live consumer. "
        "The sealed V15 compile receipt's producer_custody pins "
        "src/tac/optimization/direct_description_carrier_compose.py at 3e1f69bb/156,551 B; HEAD is "
        "6fef110d/160,470 B. Drift commits: 9934d488b then 36f4b2947 (both 2026-08-20). "
        "This is NOT a hash swap: 36f4b2947 adds key non_empty_member_payload_count to "
        "prove_carrier_archive_fail_closed, which the V15 producer embeds as "
        "receipt.fail_closed_mutation_proof, so a regenerated receipt legitimately differs. "
        "FIRE TRIGGER: a scorer-authorized arm re-runs tools/measure_ddm_v15_scorer_solved_templates.py "
        "(it solves through exact R + SegNet, which ddm_ql1's charter forbade) and either refreshes "
        "producer_custody against a bit-exact compile receipt -- then DELETE this mark -- or records "
        "real output drift. strict=True means this mark FAILS the moment the lineage is repaired. "
        "Owning memos: .omx/research/ddm_ql1_retired_lineage_test_quarantine_20260903.md and "
        ".omx/research/ddm_cd1_working_tree_debt_landing_20260903.md"
    ),
)
_SEMANTIC_SHA256 = "759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df"
_RICH_PROGRAM_SHA256 = "8bd80138198fb01d9c2a376c8918634eb0e7f08f12a15d6eb9a888dc941fb6bb"
_OPERAND_SHA256 = "5616799adc0d2ab942f37a20b070f7a0fa48119771e8f1b56c1f45e2605306ca"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _semantic() -> bytes:
    if not _SEMANTIC_PATH.is_file() or not _COMPILE_RECEIPT_PATH.is_file():
        pytest.skip("retained fresh V15 semantic custody is absent")
    payload = _SEMANTIC_PATH.read_bytes()
    assert len(payload) == 133_941
    assert _sha(payload) == _SEMANTIC_SHA256
    return payload


def _rich_program() -> TaskspaceSelectedPreimageProgramV2:
    semantic = _semantic()
    identity = verify_v15_semantic_compile_lineage(
        compile_receipt_bytes=_COMPILE_RECEIPT_PATH.read_bytes(),
        compiled_semantic_archive=semantic,
        producer_root=_ROOT,
    )
    target = ScorerTargetCustodyIdentityV1(
        target_custody_receipt_sha256=("58db7f01674c60f060a46b955fee8c4f777f31f528ebba404e871b26b17972a7"),
        target_bank_sha256=("6d2ca48ac07323c7fc3a5299023bc291363192e10130eb3bc63d446bb8e65b85"),
    )
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
    operand = RoleAwareBoundaryShearletOperandV1(
        frame_selector=SelectedPreimageFrameSelectorV1.BOTH,
        atoms=atoms,
    )
    factor = TaskspaceSelectedPreimageFactorV2(
        section_id="a.v15_roleaware_prepaint",
        source_pair_start=0,
        pair_count=600,
        operand_payload=operand.to_bytes(),
        source_receipt_sha256=("5a74c590abf2fb4bac2c7ca3a20e1be645b6e52cc95749bae55da426c73487c9"),
    )
    result = TaskspaceSelectedPreimageProgramV2(
        semantic_program_identity=identity,
        target_custody_identity=target,
        decoder_identity=V15RoleAwareDecoderIdentityV2.current(),
        compile_config=SelectedPreimageCompileConfigV2(
            source_pair_start=0,
            pair_count=600,
            maximum_packet_bytes=_MAX_PROGRAM_BYTES,
            score_budget_receipt_sha256=("5a74c590abf2fb4bac2c7ca3a20e1be645b6e52cc95749bae55da426c73487c9"),
            budget_rule_id="g82_exact_external_lowering_research_only_v1",
        ),
        factor=factor,
    )
    assert len(result.packet_bytes) == 3_593
    assert result.packet_sha256 == _RICH_PROGRAM_SHA256
    assert result.factor.operand_sha256 == _OPERAND_SHA256
    return result


def _lower(
    *,
    program: TaskspaceSelectedPreimageProgramV2 | None = None,
    semantic: bytes | None = None,
    expected_target: ScorerTargetCustodyIdentityV1 | None = None,
    expected_decoder: V15RoleAwareDecoderIdentityV2 | None = None,
) -> TSPPV2ToPVSA1LoweringV1:
    rich = _rich_program() if program is None else program
    return lower_tsppv2_to_compact_pvsa1(
        rich_program_packet=rich.packet_bytes,
        semantic_p_archive=_semantic() if semantic is None else semantic,
        expected_semantic_identity=rich.semantic_program_identity,
        expected_target_identity=(rich.target_custody_identity if expected_target is None else expected_target),
        expected_decoder_identity=(rich.decoder_identity if expected_decoder is None else expected_decoder),
        maximum_program_bytes=_MAX_PROGRAM_BYTES,
        maximum_semantic_archive_bytes=_MAX_ARCHIVE_BYTES,
        maximum_member_bytes=_MAX_MEMBER_BYTES,
        maximum_section_bytes=_MAX_SECTION_BYTES,
    )


@pytest.fixture(scope="module")
def lowering() -> TSPPV2ToPVSA1LoweringV1:
    return _lower()


@_QUARANTINE_V15_PRODUCER_PIN
def test_exact_lowering_binds_rich_operand_compact_and_baseline(
    lowering: TSPPV2ToPVSA1LoweringV1,
) -> None:
    receipt = lowering.receipt

    assert receipt.rich_program_bytes == 3_593
    assert receipt.rich_program_sha256 == _RICH_PROGRAM_SHA256
    assert receipt.rich_framing_bytes == 3_541
    assert receipt.operand_bytes == 52
    assert receipt.operand_sha256 == _OPERAND_SHA256
    assert receipt.frame_selector == "BOTH"
    assert receipt.semantic_p_bytes == 133_941
    assert receipt.semantic_p_sha256 == _SEMANTIC_SHA256
    assert receipt.semantic_baseline_member_bytes == 133_306
    assert receipt.semantic_baseline_member_sha256 == (
        "6208ac91c465caa8990f7d643f50c06da28c1e00ca359d4ee55005818cc12352"
    )
    assert receipt.semantic_baseline_archive_bytes == 129_335
    assert receipt.semantic_baseline_archive_sha256 == (
        "fa173ef4f75adbe9194d3cd89b04021dabd2b9e9fd3aa87081148b6b42a26c75"
    )
    assert receipt.semantic_container_recode_delta_bytes == -4_606
    assert receipt.compact_member_bytes == 133_363
    assert receipt.compact_member_sha256 == ("d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31")
    assert receipt.compact_archive_bytes == 129_392
    assert receipt.compact_archive_sha256 == ("b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd")
    assert receipt.actuator_same_container_marginal_bytes == 57
    assert receipt.rich_ir_bytes_avoided == 3_541
    assert (
        receipt.semantic_baseline_archive_bytes + receipt.actuator_same_container_marginal_bytes
        == receipt.compact_archive_bytes
    )


@_QUARANTINE_V15_PRODUCER_PIN
def test_no_rich_identity_or_custody_json_enters_compact_member(
    lowering: TSPPV2ToPVSA1LoweringV1,
) -> None:
    member = lowering.compact_actuated.selected.member_bytes
    rich = lowering.rich_program
    packet = lowering.rich_program_packet
    _, header_bytes = struct.Struct("<8sI").unpack_from(packet)
    header = packet[12 : 12 + header_bytes]

    assert packet not in member
    assert header not in member
    assert b"TSPPV2\x00\x00" not in member
    assert rich.target_custody_identity.target_custody_receipt_sha256.encode("ascii") not in member
    assert rich.target_custody_identity.target_bank_sha256.encode("ascii") not in member
    assert rich.decoder_identity.program_decoder_source_sha256.encode("ascii") not in member
    assert rich.decoder_identity.g74_decoder_source_sha256.encode("ascii") not in member
    assert member.count(rich.factor.operand_payload) == 1
    assert lowering.compact_actuated.selected.actuators[0].payload == (rich.factor.operand_payload)
    assert lowering.receipt.rich_identity_json_embedded_in_compact is False
    assert lowering.receipt.target_custody_embedded_in_compact is False
    assert lowering.receipt.decoder_identity_embedded_in_compact is False


@_QUARANTINE_V15_PRODUCER_PIN
def test_both_outer_encodings_and_rich_packet_strictly_parse_back(
    lowering: TSPPV2ToPVSA1LoweringV1,
) -> None:
    compact = lowering.compact_actuated
    for exact in (compact.outer_build.stored, compact.outer_build.deflated):
        outer = parse_taskspace_outer_archive(
            exact.archive_bytes,
            expected_encoding=exact.encoding,
            expected_archive_sha256=exact.archive_sha256,
            expected_member_sha256=exact.member_sha256,
            max_member_bytes=_MAX_MEMBER_BYTES,
        )
        parsed = parse_compact_pvsa_member(
            outer.member_bytes,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )
        assert parsed.member_bytes == compact.selected.member_bytes
        assert parsed.semantic_p_archive == _semantic()
        assert parsed.actuators[0].payload == lowering.rich_program.factor.operand_payload

    rich = parse_selected_preimage_program_v2(
        lowering.rich_program_packet,
        maximum_packet_bytes=_MAX_PROGRAM_BYTES,
    )
    assert rich.packet_bytes == lowering.rich_program_packet
    assert lowering.receipt.compact_store_parse_back_exact is True
    assert lowering.receipt.compact_deflate_parse_back_exact is True


@_QUARANTINE_V15_PRODUCER_PIN
def test_pair_zero_rich_and_compact_native_camera_hashes_are_equal(
    lowering: TSPPV2ToPVSA1LoweringV1,
) -> None:
    receipt = lowering.receipt

    assert receipt.rich_pair_zero_camera_sha256
    assert receipt.rich_pair_zero_camera_sha256 == (receipt.compact_pair_zero_camera_sha256)
    assert receipt.pair_zero_execution_receipt_sha256
    assert receipt.pair_zero_camera_equality_proven is True
    assert receipt.scorer_invoked is False
    assert receipt.evaluator_invoked is False


@_QUARANTINE_V15_PRODUCER_PIN
def test_external_target_decoder_and_semantic_p_drift_fail_closed() -> None:
    rich = _rich_program()
    changed_target = replace(
        rich.target_custody_identity,
        target_bank_sha256="0" * 64,
    )
    with pytest.raises(
        TSPPV2PVSA1LoweringError,
        match="external semantic/target/decoder",
    ):
        _lower(program=rich, expected_target=changed_target)

    changed_decoder = replace(
        rich.decoder_identity,
        torch_version="foreign-runtime",
    )
    with pytest.raises(
        TSPPV2PVSA1LoweringError,
        match="external semantic/target/decoder",
    ):
        _lower(program=rich, expected_decoder=changed_decoder)

    semantic = bytearray(_semantic())
    semantic[-1] ^= 1
    with pytest.raises(
        TSPPV2PVSA1LoweringError,
        match="semantic P bytes",
    ):
        _lower(program=rich, semantic=bytes(semantic))


@_QUARANTINE_V15_PRODUCER_PIN
def test_mutated_rich_packet_fails_before_lowering() -> None:
    rich = _rich_program()
    packet = bytearray(rich.packet_bytes)
    packet[-1] ^= 1
    with pytest.raises(
        TSPPV2PVSA1LoweringError,
        match="strict parse/re-emit",
    ):
        lower_tsppv2_to_compact_pvsa1(
            rich_program_packet=bytes(packet),
            semantic_p_archive=_semantic(),
            expected_semantic_identity=rich.semantic_program_identity,
            expected_target_identity=rich.target_custody_identity,
            expected_decoder_identity=rich.decoder_identity,
            maximum_program_bytes=_MAX_PROGRAM_BYTES,
            maximum_semantic_archive_bytes=_MAX_ARCHIVE_BYTES,
            maximum_member_bytes=_MAX_MEMBER_BYTES,
            maximum_section_bytes=_MAX_SECTION_BYTES,
        )


@_QUARANTINE_V15_PRODUCER_PIN
def test_external_receipt_is_canonical_self_hashable_and_truthful(
    lowering: TSPPV2ToPVSA1LoweringV1,
) -> None:
    receipt = lowering.receipt
    payload = receipt.to_bytes()
    decoded = json.loads(payload.decode("ascii"))

    assert (
        json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        == payload
    )
    assert receipt.sha256 == _sha(payload)
    assert receipt.rich_parse_reemit_exact is True
    assert receipt.operand_bytes_preserved_exactly is True
    assert receipt.public_inflate_closed is False
    assert receipt.full_n600_decode_evidence is False
    assert receipt.research_only is True
    assert receipt.candidate_claim is False
    assert receipt.score_claim is False
    assert CONDITIONAL_Y0_ACTUATOR_BLOCKER in OPEN_LOWERING_BLOCKERS
    assert PUBLIC_INFLATE_BLOCKER in lowering.open_lowering_blockers
    assert FULL_N600_BLOCKER in lowering.open_lowering_blockers
