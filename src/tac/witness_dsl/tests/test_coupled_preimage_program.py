# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import tac.witness_dsl.coupled_preimage_program as coupled_module
from tac.optimization.direct_description_carrier_compose import ReceiverRealizationProfileV1
from tac.witness_dsl.coupled_preimage_program import (
    CoupledPreimageMode,
    CoupledPreimageProgramError,
    DeclaredLineageRole,
    Frame1AnchoredY0FibreControlV1,
    JointSharedSkeletonTwoFibreControlV1,
    build_required_reference_removal_packet,
    compile_coupled_preimage_program,
    decode_coupled_preimage_program,
    parse_coupled_preimage_program,
    required_reference_removal_variants,
)
from tac.witness_dsl.generative_taskspace_correction import (
    DecodedGenerativeCorrectionV1,
    PredictorSemanticStateV1,
)


def _state() -> PredictorSemanticStateV1:
    labels = np.zeros((2, 384, 512), dtype=np.uint8)
    labels[:, 80:160] = 1
    labels[:, 160:240] = 2
    labels[:, 240:320] = 3
    labels[:, 320:] = 4
    labels[1] = np.roll(labels[1], 9, axis=1)
    pose6 = np.asarray(
        (
            (0, 1, 2, 3, 4, 5),
            (6, 7, 8, 9, 10, 11),
        ),
        dtype=np.int16,
    )
    return PredictorSemanticStateV1(
        predictor_program_sha256="a" * 64,
        predictor_renderer_sha256="b" * 64,
        source_pair_ids=(20, 21),
        labels=labels,
        pose6_codes=pose6,
    )


def _profile(*, blue_offset: int = 0) -> ReceiverRealizationProfileV1:
    return ReceiverRealizationProfileV1(
        (
            (12, 24, 36 + blue_offset),
            (48, 60, 72),
            (84, 96, 108),
            (120, 132, 144),
            (156, 168, 180),
        )
    )


def _decoded_g(
    *,
    labels: np.ndarray | None = None,
    profile: ReceiverRealizationProfileV1 | None = None,
    correction_packet_sha256: str = "c" * 64,
) -> DecodedGenerativeCorrectionV1:
    semantic_labels = _state().labels.copy() if labels is None else np.ascontiguousarray(labels, dtype=np.uint8)
    semantic_labels[0, 40:48, 40:64] = 4
    semantic_labels[1, 200:210, 100:120] = 1
    return DecodedGenerativeCorrectionV1(
        labels=semantic_labels,
        realization_profile=_profile() if profile is None else profile,
        correction_packet_sha256=correction_packet_sha256,
    )


def _anchored_controls() -> tuple[Frame1AnchoredY0FibreControlV1, ...]:
    return (
        Frame1AnchoredY0FibreControlV1(20, 1, -2, (3, -4, 5)),
        Frame1AnchoredY0FibreControlV1(21, -1, 2, (-2, 4, 1)),
    )


def _joint_controls() -> tuple[JointSharedSkeletonTwoFibreControlV1, ...]:
    return (
        JointSharedSkeletonTwoFibreControlV1(
            20,
            2,
            0,
            (
                (1, 0, 0),
                (0, 2, 0),
                (0, 0, 3),
                (-4, 0, 0),
                (0, -5, 0),
            ),
        ),
        JointSharedSkeletonTwoFibreControlV1(
            21,
            0,
            -3,
            (
                (0, 0, -1),
                (2, 0, 0),
                (0, 3, 0),
                (0, 0, 4),
                (-5, 0, 0),
            ),
        ),
    )


def _compile(mode: CoupledPreimageMode):
    kwargs = (
        {"anchored_controls": _anchored_controls()}
        if mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
        else {"joint_controls": _joint_controls()}
    )
    return compile_coupled_preimage_program(
        _state(),
        _decoded_g(),
        mode=mode,
        **kwargs,
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _rewrite_body(packet: bytes, mutation) -> bytes:
    envelope = json.loads(packet.decode("ascii"))
    mutation(envelope["body"])
    envelope["body_sha256"] = hashlib.sha256(_canonical_json(envelope["body"])).hexdigest()
    return _canonical_json(envelope)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_canonical_packet_roundtrip_reemits_exact_bytes(mode: CoupledPreimageMode) -> None:
    first = _compile(mode)
    second = _compile(mode)
    parsed = parse_coupled_preimage_program(first.packet)

    assert parsed.to_packet() == first.packet
    assert first.packet == second.packet
    assert first.receipt.packet_bytes == len(first.packet)
    assert first.receipt.packet_sha256 == hashlib.sha256(first.packet).hexdigest()
    assert first.receipt.mode is mode
    assert first.receipt.source_pairs == 2
    assert first.receipt.active_control_rows == 2
    assert first.receipt.additional_compiler_score_threshold_caps_applied is False
    assert first.receipt.expressivity_complete is False
    assert first.receipt.producer_declared_serialized_scorer_gt_oracle_e_teacher_bytes == 0
    assert first.receipt.forbidden_serialized_payload_bytes_independently_verified is False
    assert first.receipt.candidate_lineage_proven is False
    assert first.receipt.originality_proven is False
    assert first.receipt.standalone_runtime_custody is False
    assert first.receipt.research_only is True
    assert first.receipt.candidate_score_claim is False
    assert first.receipt.promotion_eligible is False
    assert first.receipt.n600_evidence_claim is False


def test_modes_are_distinct_but_both_return_exact_chronological_y0_y1() -> None:
    decoded_g = _decoded_g()
    expected_y1 = decoded_g.paint_rgb()
    anchored = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    joint = _compile(CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE)

    for compiled in (anchored, joint):
        assert np.array_equal(compiled.decoded.y1, expected_y1)
        assert np.array_equal(compiled.decoded.chronological_frames[:, 0], compiled.decoded.y0)
        assert np.array_equal(compiled.decoded.chronological_frames[:, 1], expected_y1)
        assert compiled.decoded.chronological_output_order == ("Y0", "Y1")
        assert compiled.decoded.materialization_order == (
            "G_SEMANTIC_LABELS",
            "EXACT_REALIZED_UINT8_Y1",
            "VERIFY_EXACT_Y1_CONTENT_SHA256",
            "MATERIALIZE_Y0_GIVEN_EXACT_Y1",
        )
        assert not np.array_equal(compiled.decoded.y0, compiled.decoded.y1)
        assert compiled.decoded.y0.flags.writeable is False
        assert compiled.decoded.y1.flags.writeable is False

    assert anchored.program.mode is CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE
    assert joint.program.mode is CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE
    assert anchored.program.joint_controls == ()
    assert joint.program.anchored_controls == ()
    assert not np.array_equal(anchored.decoded.y0, joint.decoded.y0)


def test_y1_hash_guard_runs_before_y0_materializer(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)

    def replace_conditioning_hash(body: dict[str, object]) -> None:
        forged_hash = "f" * 64
        source_binding = body["source_binding"]
        assert isinstance(source_binding, dict)
        source_binding["exact_y1_content_sha256"] = forged_hash
        lineage = body["lineage"]
        assert isinstance(lineage, list)
        for item in lineage:
            if item["role"] == DeclaredLineageRole.EXACT_FRAME1_UINT8.value:
                item["content_sha256"] = forged_hash

    forged_packet = _rewrite_body(compiled.packet, replace_conditioning_hash)
    called = False

    def forbidden_y0_materializer(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("Y0 materializer ran before exact Y1 hash admission")

    monkeypatch.setattr(coupled_module, "_materialize_y0", forbidden_y0_materializer)
    with pytest.raises(CoupledPreimageProgramError, match="conditioning hash mismatch before Y0"):
        decode_coupled_preimage_program(
            forged_packet,
            predictor_state=_state(),
            decoded_g=_decoded_g(),
        )
    assert called is False


def test_live_predictor_g_and_realization_hash_tampering_is_rejected() -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    changed_labels = _decoded_g().labels.copy()
    changed_labels[0, 0, 0] = np.uint8((int(changed_labels[0, 0, 0]) + 1) % 5)

    with pytest.raises(CoupledPreimageProgramError, match="correction_labels_sha256"):
        decode_coupled_preimage_program(
            compiled.packet,
            predictor_state=_state(),
            decoded_g=_decoded_g(labels=changed_labels),
        )
    with pytest.raises(CoupledPreimageProgramError, match="correction_packet_sha256"):
        decode_coupled_preimage_program(
            compiled.packet,
            predictor_state=_state(),
            decoded_g=_decoded_g(correction_packet_sha256="d" * 64),
        )
    with pytest.raises(CoupledPreimageProgramError, match="realization_profile_sha256"):
        decode_coupled_preimage_program(
            compiled.packet,
            predictor_state=_state(),
            decoded_g=_decoded_g(profile=_profile(blue_offset=1)),
        )

    changed_state = replace(_state(), predictor_renderer_sha256="e" * 64)
    with pytest.raises(CoupledPreimageProgramError, match="predictor_renderer_sha256"):
        decode_coupled_preimage_program(
            compiled.packet,
            predictor_state=changed_state,
            decoded_g=_decoded_g(),
        )


def test_decoder_binding_and_packet_bytes_are_fail_closed() -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)

    def change_decoder(body: dict[str, object]) -> None:
        binding = body["decoder_binding"]
        assert isinstance(binding, dict)
        binding["contract_sha256"] = "0" * 64

    with pytest.raises(CoupledPreimageProgramError, match="decoder contract/source binding mismatch"):
        parse_coupled_preimage_program(_rewrite_body(compiled.packet, change_decoder))

    corrupted = bytearray(compiled.packet)
    corrupted[-2] = ord("!")
    with pytest.raises(CoupledPreimageProgramError):
        parse_coupled_preimage_program(bytes(corrupted))


@pytest.mark.parametrize(
    "forbidden_role",
    (
        "SCORER_WEIGHTS",
        "GT_LABELS",
        "ORACLE_OBSERVATIONS",
        "EVALUATOR_OBLIGATION_IR",
        "TEACHER_PAYLOAD",
    ),
)
def test_forbidden_scorer_gt_oracle_e_teacher_lineage_is_rejected(forbidden_role: str) -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)

    def inject_forbidden_role(body: dict[str, object]) -> None:
        lineage = body["lineage"]
        assert isinstance(lineage, list)
        lineage[0]["role"] = forbidden_role

    with pytest.raises(CoupledPreimageProgramError, match="forbidden or outside"):
        parse_coupled_preimage_program(_rewrite_body(compiled.packet, inject_forbidden_role))


def test_closed_schema_rejects_hidden_oracle_payload_field() -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)

    def inject_oracle_payload(body: dict[str, object]) -> None:
        body["oracle_payload"] = [1, 2, 3]

    with pytest.raises(CoupledPreimageProgramError, match="closed schema"):
        parse_coupled_preimage_program(_rewrite_body(compiled.packet, inject_oracle_payload))


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_every_declared_role_has_a_parser_level_required_reference_removal(mode: CoupledPreimageMode) -> None:
    compiled = _compile(mode)
    variants = required_reference_removal_variants(compiled.packet)

    assert len(variants) == len(compiled.program.lineage)
    assert {variant.removed_role for variant in variants} == {item.role for item in compiled.program.lineage}
    assert all(item.ownership == "DECLARED_ORIGINAL_OWN_UNVERIFIED" for item in compiled.program.lineage)
    assert all(item.origin == "DECLARED_LOCAL_ORIGINAL_PROGRAM_UNVERIFIED" for item in compiled.program.lineage)
    assert all(item.candidate_lineage_proven is False for item in compiled.program.lineage)
    assert all(item.originality_proven is False for item in compiled.program.lineage)
    for variant in variants:
        assert variant.packet != compiled.packet
        assert variant.packet_sha256 == hashlib.sha256(variant.packet).hexdigest()
        envelope = json.loads(variant.packet.decode("ascii"))
        assert envelope["body_sha256"] == hashlib.sha256(_canonical_json(envelope["body"])).hexdigest()
        with pytest.raises(CoupledPreimageProgramError, match="lineage roles/content"):
            parse_coupled_preimage_program(variant.packet)
        assert variant.packet == build_required_reference_removal_packet(compiled.packet, variant.removed_role)


@pytest.mark.parametrize("mode", tuple(CoupledPreimageMode))
def test_decode_is_byte_deterministic_without_scorer_or_evidence(mode: CoupledPreimageMode) -> None:
    compiled = _compile(mode)
    first = decode_coupled_preimage_program(
        compiled.packet,
        predictor_state=_state(),
        decoded_g=_decoded_g(),
    )
    second = decode_coupled_preimage_program(
        compiled.packet,
        predictor_state=_state(),
        decoded_g=_decoded_g(),
    )

    assert first.exact_y1_content_sha256 == second.exact_y1_content_sha256
    assert first.chronological_content_sha256 == second.chronological_content_sha256
    assert np.array_equal(first.chronological_frames, second.chronological_frames)
    assert first.research_only is True
    assert first.candidate_score_claim is False


def test_behavior_quotient_rejects_universal_aliases_without_score_thresholds() -> None:
    controls = (
        Frame1AnchoredY0FibreControlV1(20, 383, -511, (255, -255, 255)),
        Frame1AnchoredY0FibreControlV1(21, -383, 511, (-255, 255, -255)),
    )
    compiled = compile_coupled_preimage_program(
        _state(),
        _decoded_g(),
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=controls,
    )
    assert compiled.receipt.additional_compiler_score_threshold_caps_applied is False
    assert compiled.receipt.control_cardinality_policy == "exactly_one_mode_specific_control_row_per_source_pair.v1"
    assert compiled.receipt.expressivity_complete is False
    assert compiled.decoded.chronological_frames.shape == (2, 2, 384, 512, 3)
    body = json.loads(compiled.packet.decode("ascii"))["body"]
    assert body["policy"]["control_behavior_quotient"].startswith("dy[-383,383]_dx[-511,511]")
    assert body["resource_counts"]["control_parameter_space_per_pair"] == str(767 * 1023 * 511**3)
    assert body["resource_counts"]["behavior_map_known_noninjective_for_specific_inputs"] is True

    with pytest.raises(CoupledPreimageProgramError, match="Y-shift quotient"):
        Frame1AnchoredY0FibreControlV1(20, 384, 0, (0, 0, 0))
    with pytest.raises(CoupledPreimageProgramError, match="X-shift quotient"):
        Frame1AnchoredY0FibreControlV1(20, 0, -512, (0, 0, 0))
    with pytest.raises(CoupledPreimageProgramError, match="uint8-delta quotient"):
        Frame1AnchoredY0FibreControlV1(20, 0, 0, (256, 0, 0))


def test_mode_control_families_and_pair_coverage_are_exact() -> None:
    with pytest.raises(CoupledPreimageProgramError, match="cannot carry joint-skeleton"):
        compile_coupled_preimage_program(
            _state(),
            _decoded_g(),
            mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
            anchored_controls=_anchored_controls(),
            joint_controls=_joint_controls(),
        )
    with pytest.raises(CoupledPreimageProgramError, match="every source pair exactly once"):
        compile_coupled_preimage_program(
            _state(),
            _decoded_g(),
            mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
            anchored_controls=_anchored_controls()[:1],
        )


def test_lineage_and_payload_classification_remain_explicitly_unverified() -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    body = json.loads(compiled.packet.decode("ascii"))["body"]
    policy = body["policy"]

    assert policy["producer_declared_forbidden_serialized_payload_bytes"] == 0
    assert policy["forbidden_serialized_payload_bytes_independently_verified"] is False
    assert policy["candidate_lineage_proven"] is False
    assert policy["originality_proven"] is False
    assert policy["expressivity_complete"] is False
    assert policy["standalone_runtime_custody"] is False

    def forge_originality(body: dict[str, object]) -> None:
        lineage = body["lineage"]
        assert isinstance(lineage, list)
        lineage[0]["originality_proven"] = True

    with pytest.raises(CoupledPreimageProgramError, match="producer-declared and unverified"):
        parse_coupled_preimage_program(_rewrite_body(compiled.packet, forge_originality))


def test_direct_decoder_source_and_expected_outputs_are_packet_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    compiled = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    body = json.loads(compiled.packet.decode("ascii"))["body"]
    source_sha = hashlib.sha256(Path(coupled_module.__file__).read_bytes()).hexdigest()

    assert body["decoder_binding"]["direct_source_sha256"] == source_sha
    assert body["decoder_binding"]["binding_scope"] == "direct_module_file_bytes_only_nontransitive.v1"
    assert body["decoder_binding"]["standalone_runtime_custody"] is False
    assert body["source_binding"]["expected_y0_content_sha256"] == compiled.receipt.expected_y0_content_sha256
    assert (
        body["source_binding"]["expected_chronological_content_sha256"] == compiled.receipt.chronological_content_sha256
    )

    original = coupled_module._materialize_y0

    def changed_materializer(*args, **kwargs):
        changed = original(*args, **kwargs).copy()
        changed[0, 0, 0, 0] = np.uint8((int(changed[0, 0, 0, 0]) + 1) % 256)
        return changed

    monkeypatch.setattr(coupled_module, "_materialize_y0", changed_materializer)
    with pytest.raises(CoupledPreimageProgramError, match="materialized Y0 differs"):
        decode_coupled_preimage_program(
            compiled.packet,
            predictor_state=_state(),
            decoded_g=_decoded_g(),
        )


def test_pose6_is_not_an_a_packet_input_or_materializer_claim() -> None:
    baseline = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    changed_pose = _state().pose6_codes.copy()
    changed_pose[:] = np.int16(-1234)
    state = replace(_state(), pose6_codes=changed_pose)
    changed = compile_coupled_preimage_program(
        state,
        _decoded_g(),
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=_anchored_controls(),
    )

    assert baseline.packet == changed.packet
    assert np.array_equal(baseline.decoded.y0, changed.decoded.y0)
    assert np.array_equal(baseline.decoded.y1, changed.decoded.y1)
    body = json.loads(changed.packet.decode("ascii"))["body"]
    assert "predictor_pose6_sha256" not in body["source_binding"]


def test_active_controls_g_and_joint_skeleton_have_real_output_counterfactuals() -> None:
    anchored = _compile(CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE)
    controls = list(_anchored_controls())
    controls[0] = replace(controls[0], shift_x_i16=controls[0].shift_x_i16 + 1)
    changed_control = compile_coupled_preimage_program(
        _state(),
        _decoded_g(),
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=tuple(controls),
    )
    assert not np.array_equal(anchored.decoded.y0, changed_control.decoded.y0)

    changed_g = compile_coupled_preimage_program(
        _state(),
        _decoded_g(profile=_profile(blue_offset=1)),
        mode=CoupledPreimageMode.FRAME1_ANCHORED_Y0_FIBRE,
        anchored_controls=_anchored_controls(),
    )
    assert not np.array_equal(anchored.decoded.y1, changed_g.decoded.y1)
    assert not np.array_equal(anchored.decoded.y0, changed_g.decoded.y0)

    exact_y1 = _decoded_g().paint_rgb()
    labels = _decoded_g().labels.copy()
    changed_labels = labels.copy()
    changed_labels[0, 100:110, 100:110] = np.uint8(2)
    base_joint_y0 = coupled_module._materialize_y0_from_controls(
        CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE,
        _state().source_pair_ids,
        (),
        _joint_controls(),
        exact_y1,
        labels,
    )
    changed_joint_y0 = coupled_module._materialize_y0_from_controls(
        CoupledPreimageMode.JOINT_SHARED_SKELETON_TWO_FIBRE,
        _state().source_pair_ids,
        (),
        _joint_controls(),
        exact_y1,
        changed_labels,
    )
    assert not np.array_equal(base_joint_y0, changed_joint_y0)
